"""Distance computation."""

import torch

from fastabx.cell import Cell
from fastabx.ed import ed_batch


def distance_on_cell(cell: Cell) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the distance matrices between all A and X, and all B and X in the ``cell``, for a given ``distance``."""
    (a, sa), (b, sb), (x, sx) = (cell.a.data, cell.a.sizes), (cell.b.data, cell.b.sizes), (cell.x.data, cell.x.sizes)
    dxa = ed_batch(x, a, sx, sa, symmetric=cell.is_symmetric)
    dxb = ed_batch(x, b, sx, sb, symmetric=False)
    return dxa, dxb


def abx_on_cell(cell: Cell) -> torch.Tensor:
    """Compute the ABX of a ``cell`` using the given ``distance``."""
    dxa, dxb = distance_on_cell(cell)
    if cell.is_symmetric:
        dxa.fill_diagonal_(dxb.max() + 1)
    nx, na = dxa.size()
    nx, nb = dxb.size()
    dxb = dxb.view(nx, 1, nb).expand(nx, na, nb)
    dxa = dxa.view(nx, na, 1).expand(nx, na, nb)
    sc = (dxa < dxb).sum() + 0.5 * (dxa == dxb).sum()
    sc /= len(cell)
    return 1 - sc
