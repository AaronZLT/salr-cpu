# source .venv/bin/activate && python salr_cpu.py

import math
import queue
import threading
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_lut() -> torch.Tensor:
    lut = torch.full((256, 8), -1, dtype=torch.long)
    for m in range(256):
        idx = 0
        for t in range(8):
            if (m >> t) & 1:
                lut[m, t] = idx
                idx += 1
    return lut


LUT_8BIT = build_lut()
POPCOUNT = torch.tensor([bin(i).count("1") for i in range(256)], dtype=torch.long)


@dataclass
class BitmapSparseWeight:
    in_features: int
    out_features: int
    masks: torch.Tensor  # [in_features, num_bytes] uint8
    offsets: torch.Tensor  # [in_features, num_bytes] long
    values: torch.Tensor  # [nnz] float

    @property
    def num_bytes(self) -> int:
        return self.masks.shape[1]

    @staticmethod
    def from_dense(weight: torch.Tensor, threshold: float) -> Tuple["BitmapSparseWeight", torch.Tensor]:
        """
        weight: [in_features, out_features]
        threshold: prune entries with abs(w) <= threshold
        """
        assert weight.dim() == 2
        in_features, out_features = weight.shape
        keep = weight.abs() > threshold
        pruned = weight * keep

        num_bytes = (out_features + 7) // 8
        masks = torch.zeros((in_features, num_bytes), dtype=torch.uint8)
        offsets = torch.zeros((in_features, num_bytes), dtype=torch.long)
        values_list: List[float] = []

        cursor = 0
        for i in range(in_features):
            for b in range(num_bytes):
                offsets[i, b] = cursor
                mask = 0
                local_vals = []
                base_col = b * 8
                for t in range(8):
                    col = base_col + t
                    if col >= out_features:
                        break
                    if keep[i, col]:
                        mask |= 1 << t
                        local_vals.append(pruned[i, col].item())
                masks[i, b] = mask
                values_list.extend(local_vals)
                cursor += len(local_vals)

        values = torch.tensor(values_list, dtype=weight.dtype)
        return BitmapSparseWeight(in_features, out_features, masks, offsets, values), pruned

    def decode_rows(self, row_start: int, row_end: int) -> torch.Tensor:
        block_rows = row_end - row_start
        dense_block = torch.zeros((block_rows, self.out_features), dtype=self.values.dtype)
        for i_local, i in enumerate(range(row_start, row_end)):
            for b in range(self.num_bytes):
                mask = int(self.masks[i, b].item())
                if mask == 0:
                    continue
                start = int(self.offsets[i, b].item())
                k = int(POPCOUNT[mask].item())
                segment = self.values[start : start + k]
                lut_row = LUT_8BIT[mask]
                base_col = b * 8
                for t in range(8):
                    col = base_col + t
                    if col >= self.out_features:
                        break
                    seg_idx = int(lut_row[t].item())
                    if seg_idx >= 0:
                        dense_block[i_local, col] = segment[seg_idx]
        return dense_block

    def decode_dense(self) -> torch.Tensor:
        return self.decode_rows(0, self.in_features)

    def matmul(self, x: torch.Tensor, row_block: int = 16, pipelined: bool = True, queue_size: int = 2) -> torch.Tensor:
        """
        Compute x @ W where W is this sparse weight (decoded in blocks).
        x: [batch, in_features]
        """
        assert x.shape[1] == self.in_features
        out = torch.zeros((x.shape[0], self.out_features), dtype=x.dtype, device=x.device)

        blocks: List[Tuple[int, int]] = []
        for s in range(0, self.in_features, row_block):
            e = min(self.in_features, s + row_block)
            blocks.append((s, e))

        if not pipelined:
            for s, e in blocks:
                w_block = self.decode_rows(s, e).to(device=x.device, dtype=x.dtype)
                out = out + x[:, s:e] @ w_block
            return out

        ring: "queue.Queue[Tuple[int, int, torch.Tensor] | None]" = queue.Queue(maxsize=max(1, queue_size))
        sentinel = None

        def decode_worker():
            for s, e in blocks:
                ring.put((s, e, self.decode_rows(s, e)))
            ring.put(sentinel)

        thread = threading.Thread(target=decode_worker, daemon=True)
        thread.start()

        while True:
            item = ring.get()
            if item is sentinel:
                break
            s, e, w_block = item
            w_block = w_block.to(device=x.device, dtype=x.dtype)
            out = out + x[:, s:e] @ w_block

        thread.join()
        return out


class ConcatenatedLowRankAdapters(nn.Module):
    """
    Implements adapter concatenation:
    sum_i (x @ A_i) @ B_i == (x @ A_cat) @ B_cat
    """

    def __init__(self, in_features: int, out_features: int, ranks: List[int], scales: List[float]):
        super().__init__()
        assert len(ranks) == len(scales)
        self.in_features = in_features
        self.out_features = out_features
        self.ranks = ranks
        self.scales = scales

        self.As = nn.ParameterList()
        self.Bs = nn.ParameterList()
        for r in ranks:
            a = nn.Parameter(torch.empty(in_features, r))
            b = nn.Parameter(torch.empty(r, out_features))
            nn.init.kaiming_uniform_(a, a=math.sqrt(5))
            nn.init.zeros_(b)
            self.As.append(a)
            self.Bs.append(b)

    def forward_sequential(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.zeros((x.shape[0], self.out_features), dtype=x.dtype, device=x.device)
        for a, b, s in zip(self.As, self.Bs, self.scales):
            out = out + (x @ a) @ (b * s)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a_cat = torch.cat(list(self.As), dim=1)
        b_cat = torch.cat([b * s for b, s in zip(self.Bs, self.scales)], dim=0)
        return (x @ a_cat) @ b_cat


def topk_threshold(weight: torch.Tensor, sparsity: float) -> float:
    assert 0.0 <= sparsity < 1.0
    n = weight.numel()
    k_prune = int(round(n * sparsity))
    if k_prune <= 0:
        return -1.0
    flat = weight.abs().flatten()
    vals, _ = torch.topk(flat, k=n - k_prune, largest=True)
    keep_min = vals[-1].item()
    return float(keep_min)


def svd_factorize(weight: torch.Tensor, rank: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns A, B such that A @ B approximates weight.
    """
    u, s, vh = torch.linalg.svd(weight, full_matrices=False)
    r = min(rank, s.shape[0])
    u_r = u[:, :r]
    s_r = s[:r]
    vh_r = vh[:r, :]
    s_sqrt = torch.sqrt(s_r)
    a = u_r * s_sqrt.unsqueeze(0)
    b = s_sqrt.unsqueeze(1) * vh_r
    return a, b


class SALRLinear(nn.Module):
    def __init__(
        self,
        dense_weight: torch.Tensor,
        bias: torch.Tensor | None,
        lora_rank: int,
        residual_rank: int,
        sparsity: float,
        row_block: int = 16,
        pipelined: bool = True,
    ):
        super().__init__()
        in_features, out_features = dense_weight.shape
        self.in_features = in_features
        self.out_features = out_features
        self.row_block = row_block
        self.pipelined = pipelined

        threshold = topk_threshold(dense_weight, sparsity)
        sparse_weight, pruned = BitmapSparseWeight.from_dense(dense_weight, threshold=threshold)
        self.sparse_weight = sparse_weight
        residual = dense_weight - pruned

        self.adapters = ConcatenatedLowRankAdapters(
            in_features=in_features,
            out_features=out_features,
            ranks=[lora_rank, residual_rank],
            scales=[1.0 / max(lora_rank, 1), 1.0],
        )

        # Initialize residual adapter (A2, B2) from truncated-SVD of pruning residual.
        with torch.no_grad():
            a2, b2 = svd_factorize(residual, rank=residual_rank)
            self.adapters.As[1][:, : a2.shape[1]].copy_(a2)
            self.adapters.Bs[1][: a2.shape[1], :].copy_(b2)

        if bias is None:
            self.bias = None
        else:
            self.bias = nn.Parameter(bias.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.sparse_weight.matmul(
            x,
            row_block=self.row_block,
            pipelined=self.pipelined,
            queue_size=2,
        )
        out = base + self.adapters(x)
        if self.bias is not None:
            out = out + self.bias
        return out


class TinySALRModel(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        w1 = torch.randn(in_dim, hidden_dim) * 0.2
        b1 = torch.zeros(hidden_dim)
        w2 = torch.randn(hidden_dim, out_dim) * 0.2
        b2 = torch.zeros(out_dim)

        self.fc1 = SALRLinear(w1, b1, lora_rank=4, residual_rank=4, sparsity=0.5, row_block=8, pipelined=True)
        self.fc2 = SALRLinear(w2, b2, lora_rank=4, residual_rank=4, sparsity=0.5, row_block=8, pipelined=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.fc1(x))
        return self.fc2(x)


def validate_components() -> None:
    torch.manual_seed(7)

    # 1) Bitmap decode correctness.
    w = torch.randn(12, 19)
    thr = topk_threshold(w, sparsity=0.5)
    sparse, pruned = BitmapSparseWeight.from_dense(w, threshold=thr)
    decoded = sparse.decode_dense()
    max_err_decode = (decoded - pruned).abs().max().item()
    print(f"[check] bitmap decode max abs err: {max_err_decode:.6g}")
    assert max_err_decode < 1e-6

    # 2) Pipeline and non-pipeline should match.
    x = torch.randn(5, 12)
    y_np = sparse.matmul(x, row_block=4, pipelined=False)
    y_pp = sparse.matmul(x, row_block=4, pipelined=True)
    max_err_pipe = (y_np - y_pp).abs().max().item()
    print(f"[check] pipeline vs non-pipeline max abs err: {max_err_pipe:.6g}")
    assert max_err_pipe < 1e-6

    # 3) Concatenated adapter equals sequential sum.
    adapters = ConcatenatedLowRankAdapters(in_features=12, out_features=19, ranks=[3, 4], scales=[0.5, 1.2])
    y_seq = adapters.forward_sequential(x)
    y_cat = adapters(x)
    max_err_cat = (y_seq - y_cat).abs().max().item()
    print(f"[check] concatenated adapters max abs err: {max_err_cat:.6g}")
    assert max_err_cat < 1e-6


def train_dummy_model() -> None:
    torch.manual_seed(42)
    n, in_dim, hidden_dim, out_dim = 512, 24, 32, 2
    x = torch.randn(n, in_dim)

    # Synthetic teacher for supervised regression target.
    teacher = nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, out_dim),
    )
    with torch.no_grad():
        y = teacher(x)

    model = TinySALRModel(in_dim, hidden_dim, out_dim)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    losses: List[float] = []
    for step in range(120):
        pred = model(x)
        loss = F.mse_loss(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if step % 20 == 0 or step == 119:
            print(f"[train] step={step:03d} loss={loss.item():.6f}")

    print(f"[result] initial_loss={losses[0]:.6f} final_loss={losses[-1]:.6f}")
    assert losses[-1] < losses[0], "Training did not reduce loss."


if __name__ == "__main__":
    validate_components()
    train_dummy_model()
    print("SALR CPU demo finished successfully.")
