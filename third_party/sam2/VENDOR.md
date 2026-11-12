# Vendored: SAM 2 (Hiera trunk only)

This directory contains a trimmed copy of the **SAM 2** source tree, used solely
as the image encoder (Hiera trunk) of the 1-shot landslide model.

- Upstream: https://github.com/facebookresearch/sam2
- License: see `LICENSE` / `LICENSE_cctorch` (Apache-2.0 / BSD-3 style; keep them).
- Only the importable Python package `sam2/` (plus licenses) is kept; the demo,
  notebooks, training, `sav_dataset` and checkpoints have been removed.

## What is used

`1-shot/model/sam2_encoder.py`

```python
from sam2.modeling.backbones.hieradet import Hiera  # via hydra instantiate
```

only the `model.image_encoder.trunk` node of a config is instantiated — the
prompt encoder, mask decoder, memory modules and FPN neck are never built.

## Config resolution

`sam2/__init__.py` calls `initialize_config_module("sam2")`, so hydra composes
configs by *name*. The names on disk are:

    sam2/configs/sam2/sam2_hiera_{t,s,b+,l}.yaml

The root-level shim files (`sam2/sam2_hiera_l.yaml` containing
`configs/sam2/sam2_hiera_l.yaml`) are **not** real hydra configs, so
`sam2_encoder._compose_trunk_cfg()` transparently retries the full
`configs/sam2/...` path when a bare `sam2_hiera_l.yaml` yields no `model` node.

## Checkpoint

Download the **SAM 2.0** weights (not 2.1) from upstream, e.g.:

    curl -L -o sam2_hiera_large.pt \
      https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt

and pass it via `SAM2_CKPT=/path/to/sam2_hiera_large.pt` (or `--sam2-ckpt`).
Only the `image_encoder.trunk.*` tensors are read from it.

## Runtime dependencies

The package needs `hydra-core`, `omegaconf` and `iopath` (see `requirements.txt`);
`pip install sam2` is **not** required or wanted.
