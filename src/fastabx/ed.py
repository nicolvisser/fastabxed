"""Edit distance implementation using PyTorch C++ extensions, with CPU and CUDA backends."""

import torch


def ed(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute the DTW of the given ``distances`` 2D tensor."""
    return torch.ops.fastabx.ed.default(x, y)


def ed_batch(x: torch.Tensor, y: torch.Tensor, sx: torch.Tensor, sy: torch.Tensor, *, symmetric: bool) -> torch.Tensor:
    """Compute the batched DTW on the ``distances`` 4D tensor."""
    return torch.ops.fastabx.ed_batch.default(x, y, sx, sy, symmetric)


@torch.library.register_fake("fastabx::ed")
def _(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Register the FakeTensor kernel for ed, for compatibility with torch.compile."""
    torch._check(x.ndim == 1)  # noqa: PLR2004
    torch._check(y.ndim == 1)
    torch._check(x.dtype == torch.long)
    torch._check(y.dtype == torch.long)
    return torch.empty((), dtype=torch.long, layout=x.layout, device=x.device)


@torch.library.register_fake("fastabx::ed_batch")
def _(x: torch.Tensor, y: torch.Tensor, sx: torch.Tensor, sy: torch.Tensor, symmetric: bool) -> torch.Tensor:  # noqa: FBT001, ARG001
    """Register the FakeTensor kernel for ed_batch, for compatibility with torch.compile."""
    torch._check(x.ndim == 2)
    torch._check(y.ndim == 2)
    torch._check(sx.ndim == 1)
    torch._check(sy.ndim == 1)
    torch._check(x.dtype == torch.long)
    torch._check(y.dtype == torch.long)
    torch._check(sx.dtype == torch.long)
    torch._check(sy.dtype == torch.long)
    nx, _ = x.shape
    ny, _ = y.shape
    return torch.empty((nx, ny), dtype=torch.long, layout=x.layout, device=x.device)
