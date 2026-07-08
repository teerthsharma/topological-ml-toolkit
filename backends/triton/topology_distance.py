"""Optional Triton kernels for topology preprocessing.

This module is not imported by ``topoml``. Import it only inside explicit
Triton benchmark or backend experiments so the core package keeps its optional
dependency contract.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _pairwise_l2_kernel(points, out, n_points: tl.constexpr, dim: tl.constexpr, block: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, block)
    offsets = cols
    acc = tl.zeros((block,), dtype=tl.float32)
    mask = cols < n_points
    for k in range(0, dim):
        row_value = tl.load(points + row * dim + k)
        col_value = tl.load(points + offsets * dim + k, mask=mask, other=0.0)
        delta = row_value - col_value
        acc += delta * delta
    tl.store(out + row * n_points + offsets, tl.sqrt(acc), mask=mask)


def pairwise_l2(points: torch.Tensor, block: int = 1024) -> torch.Tensor:
    if not points.is_cuda:
        raise ValueError("points must be a CUDA tensor")
    if points.ndim != 2:
        raise ValueError("points must be a 2D tensor")
    if points.dtype not in (torch.float16, torch.float32):
        raise ValueError("points must be float16 or float32")
    n_points, dim = points.shape
    if n_points > block:
        raise ValueError("prototype Triton kernel currently requires n_points <= block")
    out = torch.empty((n_points, n_points), device=points.device, dtype=torch.float32)
    _pairwise_l2_kernel[(n_points,)](points, out, n_points=n_points, dim=dim, block=block)
    return out
