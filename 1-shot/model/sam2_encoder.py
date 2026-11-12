r""" SAM2 Hiera image encoder (encoder-only) with parameter-efficient adapters.

Replaces the torchvision ``resnet``/``vgg`` encoder inside :class:`IFCNet` while
keeping the fusion + decoder untouched.  Only the Hiera *trunk* of SAM2 is kept
(the prompt encoder, mask decoder, memory modules and the FPN neck are never
built), and the trunk is frozen; a small :class:`Adapter` is inserted into every
Hiera block so a handful of extra parameters can be trained.

Follows the encoder design of SAM2-UNet (https://github.com/WZH0120/SAM2-UNet):

    x1, x2, x3, x4 = trunk(x)          # strided 4, 8, 16, 32
    features = [x2, x3, x4]            # x1 dropped, fine -> coarse

``forward`` therefore returns a **3-element list** in fine->coarse order so that
``IFCNet`` (with ``nbottlenecks = [1, 1, 1, 1]``) groups it into exactly three
levels.  Channel layout per variant (Hiera ``embed_dim``):

    =================  ======================  =========================
    variant            ``x1, x2, x3, x4``        ``multilevel_ch``
    =================  ======================  =========================
    ``hiera_t``        96, 192, 384, 768        ``[192, 384, 768]``
    ``hiera_s``        96, 192, 384, 768        ``[192, 384, 768]``
    ``hiera_b+``       112, 224, 448, 896       ``[224, 448, 896]``
    ``hiera_l``        144, 288, 576, 1152      ``[288, 576, 1152]``
    =================  ======================  =========================

Input normalization matches SAM2 (ImageNet mean/std), i.e. the same preprocessing
the landslide loader already applies to the RGB modality.
"""
import os
import sys
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

# --------------------------------------------------------------------------- #
# make the vendored SAM2 package importable (`third_party/sam2/sam2`)
# --------------------------------------------------------------------------- #
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))          # <repo>/1-shot/model
_SHOT_DIR = os.path.dirname(_THIS_DIR)                          # <repo>/1-shot
_REPO_DIR = os.path.dirname(_SHOT_DIR)                          # <repo>
_SAM2_ROOT = os.environ.get("SAM2_ROOT") or os.path.join(_REPO_DIR, "third_party", "sam2")
if os.path.isdir(_SAM2_ROOT) and _SAM2_ROOT not in sys.path:
    sys.path.insert(0, _SAM2_ROOT)

import sam2  # noqa: F401  (also runs `initialize_config_module("sam2")`)
from hydra import compose
from hydra.utils import instantiate
from omegaconf import OmegaConf


def _compose_trunk_cfg(config_file: str):
    """Compose the Hydra config and return the ``image_encoder.trunk`` node.

    ``config_file`` may be given as ``sam2_hiera_l.yaml`` (SAM2-UNet style) or as
    an explicit ``configs/sam2/sam2_hiera_l.yaml`` path; both are tried, plus the
    2.1 tree, so the vendored repo layout does not matter.
    """
    name = str(config_file)
    if not name.endswith(".yaml"):
        name += ".yaml"

    candidates = [name]
    if "/" not in name:
        candidates += [
            os.path.join("configs", "sam2", name),
            os.path.join("configs", "sam2.1", name),
        ]
    else:
        # allow a bare "configs/..." to fall back to the repo default tree
        candidates += [os.path.join("configs", "sam2", os.path.basename(name))]

    last_err = None
    for cand in candidates:
        try:
            cfg = compose(config_name=cand)
            OmegaConf.resolve(cfg)
            trunk_cfg = cfg.model.image_encoder.trunk
            return trunk_cfg
        except Exception as exc:  # noqa: BLE001 - report the full list on failure
            last_err = exc
    raise RuntimeError(
        f"Could not compose SAM2 trunk config from {config_file!r} "
        f"(tried: {candidates}). Last error: {type(last_err).__name__}: {last_err}"
    )


def _load_trunk_checkpoint(trunk: nn.Module, ckpt_path: str) -> None:
    """Load Hiera trunk weights from a SAM2 checkpoint (``image_encoder.trunk.*``)."""
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]

    prefix = "image_encoder.trunk."
    filtered = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    if not filtered:
        raise RuntimeError(
            f"No '{prefix}*' weights found in checkpoint {ckpt_path!r}. "
            "Expected a SAM2 (2.0) checkpoint such as sam2_hiera_large.pt."
        )

    missing, unexpected = trunk.load_state_dict(filtered, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys while loading SAM2 trunk: {unexpected}")
    real_missing = [k for k in missing if "pos_embed" not in k]
    if real_missing:
        raise RuntimeError(f"Missing keys while loading SAM2 trunk: {real_missing}")


class Adapter(nn.Module):
    """Bottleneck residual adapter wrapping a frozen Hiera block.

    A 2-layer MLP (dim -> 32 -> dim) produces a prompt that is added to the block
    input before the (frozen) block is applied.  Only the MLP is trainable.
    """

    def __init__(self, block: nn.Module, bottleneck: int = 32) -> None:
        super().__init__()
        self.block = block
        dim = block.attn.qkv.in_features
        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prompt = self.prompt_learn(x)
        return self.block(x + prompt)


class SAM2Encoder(nn.Module):
    """SAM2 Hiera trunk + adapters, returning 3 feature maps (fine -> coarse)."""

    #: variant -> (x1, x2, x3, x4) channels
    CHANNELS = {
        "sam2_hiera_t": (96, 192, 384, 768),
        "sam2_hiera_s": (96, 192, 384, 768),
        "sam2_hiera_b+": (112, 224, 448, 896),
        "sam2_hiera_l": (144, 288, 576, 1152),
    }

    def __init__(
        self,
        config_file: str = "sam2_hiera_l.yaml",
        ckpt_path: Optional[str] = None,
        adapters: bool = True,
        bottleneck: int = 32,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.config_file = config_file
        self.ckpt_path = ckpt_path

        trunk_cfg = _compose_trunk_cfg(config_file)
        trunk = instantiate(trunk_cfg)

        if ckpt_path and str(ckpt_path).lower() not in ("none", "null", ""):
            if not os.path.isfile(ckpt_path):
                raise FileNotFoundError(f"SAM2 checkpoint not found: {ckpt_path}")
            _load_trunk_checkpoint(trunk, ckpt_path)

        # freeze the pretrained trunk
        for p in trunk.parameters():
            p.requires_grad = False

        # insert a trainable adapter into every block
        self.use_adapters = bool(adapters)
        if self.use_adapters:
            trunk.blocks = nn.ModuleList(
                [Adapter(blk, bottleneck=bottleneck) for blk in trunk.blocks]
            )

        self.trunk = trunk
        self.to(device)

    # ------------------------------------------------------------------ props
    @property
    def out_channels(self) -> Tuple[int, int, int]:
        """Channels of the 3 returned maps (x2, x3, x4), fine -> coarse."""
        ch = list(self.trunk.channel_list)          # [x4, x3, x2, x1]
        return (ch[2], ch[1], ch[0])

    @property
    def strides(self) -> Tuple[int, int, int]:
        return (8, 16, 32)

    # ------------------------------------------------------------------ utils
    def train(self, mode: bool = True):
        super().train(mode)
        # Hiera must stay in eval: SAM2 ships eval-only weights and the trunk has
        # stochastic-depth (DropPath).  Adapters contain no BN/Dropout, so keeping
        # only the trunk in eval does not change adapter behaviour.
        self.trunk.eval()
        return self

    def requires_grad_features(self) -> bool:
        """Whether the extracted features keep the graph (adapters trainable)."""
        return self.use_adapters

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        if self.use_adapters:
            outputs = self.trunk(x)
        else:
            with torch.no_grad():
                outputs = self.trunk(x)
        # outputs: [x1, x2, x3, x4] (stride 4, 8, 16, 32) -> drop x1
        return outputs[1:]
