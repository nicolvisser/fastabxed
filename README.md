# Fast ABX evaluation (with edit distance)

This repo, is a fork of [fastabx](https://github.com/bootphon/fastabx) - a Python package for efficient computation of ABX discriminability.

Instead of evaluating with dynamic time warping (DTW) on vector sequences we evaluate with the normalized edit distance (ED) on discrete sequences.
We also remove consecutive duplicate IDs from the sequences corresponding to the phone segments before the ED computation.

Concretely, comparing a phone segment with ID sequence `[6, 6, 6, 1, 1, 7, 7]` to one with `[6, 6, 7, 7, 7]` is the same as comparing `[6, 1, 7]` to `[6, 7]`. It gives an ED of `1` (a single insertion) and therefore a normalized ED of `1/max(3,2)` = `0.3333`.

## Install

Build and install the package in your environment:

```bash
git clone https://github.com/nicolvisser/fastabxed
cd fastabxed
pip install -e .
```

It requires Python 3.12 or later and the default PyTorch version on PyPI (2.7.1, CUDA 12.6 variant for Linux, CPU variant for Windows and macOS).

## Usage

```py
import torch
import numpy as np

from fastabx import zerospeech_abx

def feature_maker(path: str):
    return torch.from_numpy(np.load(path)) # long tensor shape (N,) with discrete IDs

result = zerospeech_abx(
    item="path/to/item/file/phoneme-dev-clean.item",
    root="path/to/units/for/LibriSpeech/dev-clean",
    speaker="within",
    context="any",
    frequency=50,
    feature_maker=feature_maker,
    extension=".npy",
)

print(result)
```

## Citation

If you use this fork in your work, please consider citing the original work on [fastabx](https://github.com/bootphon/fastabx):

Their preprint is available on arXiv: https://arxiv.org/abs/2505.02692

```bibtex
@misc{fastabx,
  title={fastabx: A library for efficient computation of ABX discriminability},
  author={Maxime Poli and Emmanuel Chemla and Emmanuel Dupoux},
  year={2025},
  eprint={2505.02692},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2505.02692},
}
```
