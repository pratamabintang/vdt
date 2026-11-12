r""" Few-shot landslide episode dataset (RGB + DTM).

Layout expected under ``data_dir``::

    <data_dir>/<split>/<IMAGE>/<name>.png
    <data_dir>/<split>/<DTM>/<name>.tif
    <data_dir>/<split>/<LABEL>/<name>.png

Folder names are resolved case-insensitively and by alias, so both of these work::

    image / dtm / label
    IMAGE / DTM_NORM / LABEL

(file stems are also matched case-insensitively, e.g. ``Chainage_1_00026`` vs
``CHAINAGE_1_00026``.)

Alternatively ``data_dir`` may point straight at a single split folder (one that
already contains the modality folders); both train and val (and test) paths can
therefore live in separate locations on disk.

Only landslide-positive tiles (label has at least one foreground pixel) are used
for both query and support, per the project setting. Preprocessing mirrors the
supervised loader in ``datasets/landslide/dataset.py``:
  * optical RGB  : photometric aug (train) + ImageNet normalization
  * DTM          : per-tile min-max -> [0,1], replicated to 3 channels, normalized
  * label        : binarized (>0 -> 1)
  * geometry     : landslide-preserving crop + flips (train) then resize to ``size``
"""
import os
import random
from typing import Optional

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset
from PIL import Image

try:
    import cv2
    try:
        cv2.setLogLevel(0)
    except Exception:
        pass
except Exception:  # cv2 optional
    cv2 = None


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DTM_MEAN = [0.5, 0.5, 0.5]
DTM_STD = [0.5, 0.5, 0.5]


def _norm_key(name: str) -> str:
    """Lower-case + strip non-alphanumerics: DTM_NORM == dtm-norm == dtmnorm."""
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def load_blacklist(blacklist_path: Optional[str]) -> set:
    """Loads sample stems to ignore from a blacklist text file with fallback search."""
    if not blacklist_path or str(blacklist_path).lower() in ("none", "null", ""):
        return set()

    resolved_path = None
    if os.path.exists(blacklist_path):
        resolved_path = blacklist_path
    else:
        module_dir = os.path.dirname(os.path.abspath(__file__))            # <repo>/1-shot/data
        shot_dir = os.path.dirname(module_dir)                             # <repo>/1-shot
        repo_dir = os.path.dirname(shot_dir)                               # <repo>
        candidates = [
            os.path.join(module_dir, str(blacklist_path)),
            os.path.join(shot_dir, str(blacklist_path)),
            os.path.join(repo_dir, str(blacklist_path)),
            os.path.join(os.getcwd(), str(blacklist_path)),
            os.path.join(repo_dir, "datasets", "landslide", "black_list.txt"),
            os.path.join("datasets", "landslide", "black_list.txt"),
        ]
        for c in candidates:
            if os.path.exists(c):
                resolved_path = c
                break

    if not resolved_path or not os.path.exists(resolved_path):
        return set()

    blacklist = set()
    with open(resolved_path, "r", encoding="utf-8") as f:
        for line in f:
            clean = line.strip()
            if not clean or clean.startswith("#"):
                continue
            clean = clean.split("#")[0].strip()
            if not clean:
                continue
            stem, _ = os.path.splitext(os.path.basename(clean))
            blacklist.add(stem)
            blacklist.add(stem.lower())
            blacklist.add(clean)
            blacklist.add(clean.lower())
    return blacklist


def _read_raster(path: str) -> Optional[np.ndarray]:
    """Reads a single-band raster as float32, trying PIL, cv2 then tifffile."""
    try:
        with Image.open(path) as img:
            arr = np.array(img)
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        return arr.astype(np.float32)
    except Exception:
        pass

    if cv2 is not None:
        try:
            cv_img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if cv_img is not None:
                if cv_img.ndim == 3:
                    cv_img = cv_img[:, :, 0]
                return cv_img.astype(np.float32)
        except Exception:
            pass

    try:
        import tifffile
        arr = tifffile.imread(path)
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        return arr.astype(np.float32)
    except Exception:
        return None


class LandslideFSS(Dataset):
    """Few-shot landslide episodes with RGB + DTM modalities."""

    def __init__(
        self,
        data_dir: str = "../datasets/landslide",
        split: str = "train",
        size: int = 512,
        blacklist_path: Optional[str] = "../datasets/landslide/black_list.txt",
        mode: Optional[str] = None,
        min_foreground: int = 1,
        episodes_per_epoch: int = 0,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.target_size = size
        self.mode = mode if mode is not None else ("train" if split == "train" else "val")
        self.min_foreground = min_foreground

        self.split_dir = os.path.join(data_dir, split)
        if not os.path.isdir(self.split_dir):
            self.split_dir = data_dir
        if not os.path.isdir(self.split_dir):
            raise FileNotFoundError(f"Split directory not found: data_dir='{data_dir}', split='{split}'")

        self.image_dir = self._resolve_subdir(self.split_dir, ("image", "images", "rgb"))
        self.dtm_dir = self._resolve_subdir(self.split_dir, ("dtm_norm", "dtm", "dtmnormalized", "dem", "dsm"))
        self.label_dir = self._resolve_subdir(self.split_dir, ("label", "labels", "mask", "masks", "gt"))

        missing = [n for n, d in (("image", self.image_dir), ("dtm", self.dtm_dir), ("label", self.label_dir)) if d is None]
        if missing:
            found = sorted(d for d in os.listdir(self.split_dir) if os.path.isdir(os.path.join(self.split_dir, d)))
            raise FileNotFoundError(
                f"Required modality dir(s) missing under '{self.split_dir}': {missing}. "
                f"Expected e.g. IMAGE / DTM_NORM / LABEL. Found dirs: {found}"
            )

        # case-insensitive file lookup: normalized stem -> actual filename
        self._image_files = self._index_files(self.image_dir, (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"))
        self._dtm_files = self._index_files(self.dtm_dir, (".tif", ".tiff", ".png", ".bmp"))
        self._label_files = self._index_files(self.label_dir, (".png", ".tif", ".tiff", ".bmp"))

        self.blacklist = load_blacklist(blacklist_path)

        skipped_blacklist = 0
        self.samples = []
        self.pos_samples = []
        for key, f in sorted(self._image_files.items()):
            if key in self.blacklist:
                skipped_blacklist += 1
                continue
            label_path = self._label_path(f)
            if label_path is None:
                continue
            if self._count_foreground(label_path) >= self.min_foreground:
                self.pos_samples.append(f)
            self.samples.append(f)

        if skipped_blacklist > 0:
            print(f"[{split.upper()}] Blacklist active: {skipped_blacklist} broken samples skipped.")

        if len(self.pos_samples) == 0:
            raise RuntimeError(
                f"No landslide-positive tiles found in {self.split_dir} "
                f"(total: {len(self.samples)}, blacklisted: {skipped_blacklist})."
            )

        # Episodes per epoch. 0 -> one pass over all positive tiles (deterministic
        # coverage); >0 -> that many randomly-sampled query episodes each epoch.
        self.episodes_per_epoch = int(episodes_per_epoch) if episodes_per_epoch else len(self.pos_samples)

    # ------------------------------------------------------------------ paths
    @staticmethod
    def _resolve_subdir(parent: str, candidates) -> Optional[str]:
        """Find a modality subfolder by exact/case-insensitive/normalized name."""
        entries = [d for d in os.listdir(parent) if os.path.isdir(os.path.join(parent, d))]
        by_lower = {d.lower(): d for d in entries}
        for cand in candidates:
            if cand in by_lower:
                return os.path.join(parent, by_lower[cand])
        cand_norm = {_norm_key(c) for c in candidates}
        for d in entries:
            if _norm_key(d) in cand_norm:
                return os.path.join(parent, d)
        return None

    @staticmethod
    def _index_files(directory: str, exts) -> dict:
        """Map normalized stems -> actual filename (case-insensitive lookup)."""
        exts = tuple(e.lower() for e in exts)
        index = {}
        for f in os.listdir(directory):
            stem, ext = os.path.splitext(f)
            if ext.lower() in exts:
                index.setdefault(_norm_key(stem), f)
        return index

    def _label_path(self, name: str) -> Optional[str]:
        f = self._label_files.get(_norm_key(os.path.splitext(os.path.basename(name))[0]))
        return os.path.join(self.label_dir, f) if f else None

    def _image_path(self, name: str) -> Optional[str]:
        f = self._image_files.get(_norm_key(os.path.splitext(os.path.basename(name))[0]))
        return os.path.join(self.image_dir, f) if f else None

    def _dtm_path(self, name: str) -> Optional[str]:
        f = self._dtm_files.get(_norm_key(os.path.splitext(os.path.basename(name))[0]))
        return os.path.join(self.dtm_dir, f) if f else None

    @staticmethod
    def _count_foreground(label_path: str) -> int:
        try:
            with Image.open(label_path) as img:
                arr = np.array(img)
        except Exception:
            arr = _read_raster(label_path)
            if arr is None:
                return 0
        return int((arr > 0).sum())

    def __len__(self) -> int:
        # number of query episodes per epoch
        return self.episodes_per_epoch

    # ------------------------------------------------------------- loaders
    def _apply_photometric_augmentations(self, rgb: np.ndarray) -> np.ndarray:
        """Color / texture augmentations applied EXCLUSIVELY to optical RGB."""
        if random.random() < 0.5:
            contrast_factor = random.uniform(0.7, 1.3)
            brightness_offset = random.uniform(-51.0, 51.0)
            rgb = np.clip(
                contrast_factor * rgb.astype(np.float32) + brightness_offset, 0, 255
            ).astype(np.uint8)

            if cv2 is not None:
                hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
                sat_factor = random.uniform(0.8, 1.2)
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_factor, 0, 255)
                rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        if cv2 is not None and random.random() < 0.3:
            lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        if cv2 is not None and random.random() < 0.2:
            k = random.choice([3, 5])
            if random.random() < 0.5:
                rgb = cv2.GaussianBlur(rgb, (k, k), 0)
            else:
                kernel = np.zeros((k, k), dtype=np.float32)
                if random.random() < 0.5:
                    kernel[int((k - 1) / 2), :] = np.ones(k, dtype=np.float32)
                else:
                    kernel[:, int((k - 1) / 2)] = np.ones(k, dtype=np.float32)
                kernel /= k
                rgb = cv2.filter2D(rgb, -1, kernel)

        return rgb

    def _load_rgb(self, name: str) -> torch.Tensor:
        path = self._image_path(name)
        if path is None:
            raise FileNotFoundError(f"IMAGE not found for sample '{name}' in {self.image_dir}")
        try:
            with Image.open(path) as img:
                rgb = np.array(img.convert("RGB"), dtype=np.uint8)
        except Exception:
            if cv2 is not None:
                cv_img = cv2.imread(path)
                rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB) if cv_img is not None else \
                    np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)
            else:
                rgb = np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)

        if self.mode == "train":
            rgb = self._apply_photometric_augmentations(rgb)

        tensor = TF.to_tensor(rgb)  # (3,H,W) in [0,1]
        return TF.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def _load_dtm(self, name: str) -> torch.Tensor:
        path = self._dtm_path(name)
        if path is None:
            raise FileNotFoundError(f"DTM not found for sample '{name}' in {self.dtm_dir}")
        arr = _read_raster(path)
        if arr is None:
            arr = np.zeros((self.target_size, self.target_size), dtype=np.float32)

        if np.isnan(arr).any() or np.isinf(arr).any():
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        # Per-tile min-max normalization over valid elevation range
        valid_mask = (arr > -1000.0) & (arr < 10000.0)
        if valid_mask.any():
            val_min = float(arr[valid_mask].min())
            val_max = float(arr[valid_mask].max())
        else:
            val_min, val_max = 0.0, 0.0
        if val_max > val_min:
            norm = np.clip((arr - val_min) / (val_max - val_min), 0.0, 1.0)
        else:
            norm = np.zeros_like(arr)

        tensor = torch.from_numpy(norm.astype(np.float32)).unsqueeze(0)  # (1,H,W)
        tensor = tensor.repeat(3, 1, 1)  # backbone expects 3 channels
        return TF.normalize(tensor, mean=DTM_MEAN, std=DTM_STD)

    def _load_label(self, name: str) -> torch.Tensor:
        path = self._label_path(name)
        if path is None:
            raise FileNotFoundError(f"LABEL not found for sample '{name}' in {self.label_dir}")
        try:
            with Image.open(path) as img:
                arr = np.array(img)
        except Exception:
            arr = _read_raster(path)
            if arr is None:
                arr = np.zeros((self.target_size, self.target_size), dtype=np.float32)
        bin_arr = (arr > 0).astype(np.float32)
        return torch.from_numpy(bin_arr).unsqueeze(0)  # (1,H,W)

    # ------------------------------------------------------------- geometry
    def _get_landslide_preserving_crop_params(self, label: torch.Tensor,
                                              scale_range=(0.8, 0.95)):
        _, H, W = label.shape
        scale = random.uniform(*scale_range)
        ch = int(H * scale)
        cw = int(W * scale)

        pos_idx = torch.nonzero(label.squeeze(0) > 0)
        if len(pos_idx) > 0:
            y_min = int(pos_idx[:, 0].min().item())
            y_max = int(pos_idx[:, 0].max().item())
            x_min = int(pos_idx[:, 1].min().item())
            x_max = int(pos_idx[:, 1].max().item())

            cy = (y_min + y_max) // 2
            cx = (x_min + x_max) // 2

            y_low = max(0, y_max - ch)
            y_high = min(H - ch, y_min)
            ideal_y = cy - ch // 2
            if y_low <= y_high:
                y0 = max(y_low, min(ideal_y, y_high))
            else:
                y0 = max(0, min(ideal_y, H - ch))

            x_low = max(0, x_max - cw)
            x_high = min(W - cw, x_min)
            ideal_x = cx - cw // 2
            if x_low <= x_high:
                x0 = max(x_low, min(ideal_x, x_high))
            else:
                x0 = max(0, min(ideal_x, W - cw))
        else:
            y0 = random.randint(0, max(0, H - ch))
            x0 = random.randint(0, max(0, W - cw))

        return y0, x0, ch, cw

    def _geom_aug(self, rgb: torch.Tensor, dtm: torch.Tensor, mask: torch.Tensor):
        # synced geometric augs on the whole episode
        if self.mode == "train" and random.random() < 0.8:
            y0, x0, ch, cw = self._get_landslide_preserving_crop_params(mask)
            rgb = TF.crop(rgb, y0, x0, ch, cw)
            dtm = TF.crop(dtm, y0, x0, ch, cw)
            mask = TF.crop(mask, y0, x0, ch, cw)

        flip_h = self.mode == "train" and random.random() < 0.5
        flip_v = self.mode == "train" and random.random() < 0.5
        if flip_h:
            rgb, dtm, mask = TF.hflip(rgb), TF.hflip(dtm), TF.hflip(mask)
        if flip_v:
            rgb, dtm, mask = TF.vflip(rgb), TF.vflip(dtm), TF.vflip(mask)

        rgb = TF.resize(rgb, [self.target_size, self.target_size],
                        interpolation=InterpolationMode.BILINEAR, antialias=True)
        dtm = TF.resize(dtm, [self.target_size, self.target_size],
                        interpolation=InterpolationMode.BILINEAR, antialias=True)
        mask = TF.resize(mask, [self.target_size, self.target_size],
                         interpolation=InterpolationMode.NEAREST)
        return rgb, dtm, mask

    # ------------------------------------------------------------- indexing
    def __getitem__(self, idx: int):
        # deterministic coverage when episodes == #positives, else random episode
        if self.episodes_per_epoch == len(self.pos_samples):
            query_name = self.pos_samples[idx % len(self.pos_samples)]
        else:
            query_name = random.choice(self.pos_samples)

        # shared geometric parameters -> load both query and support, then aug together
        query_rgb = self._load_rgb(query_name)
        query_dtm = self._load_dtm(query_name)
        query_mask = self._load_label(query_name)

        # support: another positive tile from the same split
        if len(self.pos_samples) > 1:
            support_name = random.choice(self.pos_samples)
            while support_name == query_name:
                support_name = random.choice(self.pos_samples)
        else:
            support_name = query_name
        support_rgb = self._load_rgb(support_name)
        support_dtm = self._load_dtm(support_name)
        support_mask = self._load_label(support_name)

        # independent augs for query and support (standard FSS episode)
        query_rgb, query_dtm, query_mask = self._geom_aug(query_rgb, query_dtm, query_mask)
        support_rgb, support_dtm, support_mask = self._geom_aug(support_rgb, support_dtm, support_mask)

        subcls = 0  # single foreground class: landslide
        return query_rgb, query_dtm, query_mask, support_rgb, support_dtm, support_mask, subcls
