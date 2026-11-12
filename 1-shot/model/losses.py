r""" Loss functions for few-shot landslide segmentation.

Landslide is a heavily imbalanced binary task (foreground ~2%).  A plain
(weighted) BCE is driven by the 98% background and lets the model settle on the
trivial "predict background everywhere" solution.  The losses here therefore all
provide a foreground-aware region term (Dice / Tversky) and/or a re-balanced
pixel term (weighted BCE / focal).

Use :func:`segmentation_loss` and pick the variant with ``--loss``:

    structure   weighted BCE + weighted IoU (SINet/PraNet style; legacy default)
    dice        Dice + weighted BCE
    tversky     Tversky (FN-penalising) + weighted BCE     <- default
    focal       Focal BCE + Dice
    bce         weighted BCE only

All functions accept ``pred`` as either ``(B,2,H,W)`` (2-class classifier; the
foreground logit ``pred[:,1:2]`` is used) or ``(B,1,H,W)`` foreground logits, and
``mask`` as ``(B,H,W)`` or ``(B,1,H,W)`` with values in {0,1}.
"""
import torch
import torch.nn.functional as F

LOSS_KINDS = ("structure", "dice", "tversky", "focal", "bce")


def _split(pred, mask):
    """Return (fg_logits (B,1,H,W), mask (B,1,H,W) float)."""
    if pred.dim() == 4 and pred.size(1) == 2:
        pred = pred[:, 1:2]                # foreground logits only
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    return pred, mask.float()


def _pos_weight(pos_weight, pred):
    if pos_weight is None:
        return None
    if isinstance(pos_weight, torch.Tensor):
        return pos_weight.to(device=pred.device, dtype=pred.dtype)
    return torch.as_tensor(float(pos_weight), device=pred.device, dtype=pred.dtype)


def structure_loss(pred, mask):
    r"""Weighted BCE + weighted IoU loss (structure-aware, SINet/PraNet style)."""
    pred, mask = _split(pred, mask)

    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    prob = torch.sigmoid(pred)
    inter = ((prob * mask) * weit).sum(dim=(2, 3))
    union = ((prob + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def bce_loss(pred, mask, pos_weight=None):
    r"""Balanced BCE; ``pos_weight`` re-weights the rare foreground class."""
    pred, mask = _split(pred, mask)
    return F.binary_cross_entropy_with_logits(pred, mask, pos_weight=_pos_weight(pos_weight, pred))


def dice_loss(pred, mask, eps=1.0):
    r"""Soft Dice on the foreground.  Denominator = fg-relevant pixels only, so
    the gradient is *not* diluted by the 98% background."""
    pred, mask = _split(pred, mask)
    prob = torch.sigmoid(pred)
    inter = (prob * mask).sum(dim=(1, 2, 3))
    denom = prob.sum(dim=(1, 2, 3)) + mask.sum(dim=(1, 2, 3))
    return (1 - (2 * inter + eps) / (denom + eps)).mean()


def tversky_loss(pred, mask, alpha=0.3, beta=0.7, eps=1.0):
    r"""Tversky loss.  ``beta`` > ``alpha`` penalises false negatives (missed
    landslide) more than false positives."""
    pred, mask = _split(pred, mask)
    prob = torch.sigmoid(pred)
    tp = (prob * mask).sum(dim=(1, 2, 3))
    fp = (prob * (1 - mask)).sum(dim=(1, 2, 3))
    fn = ((1 - prob) * mask).sum(dim=(1, 2, 3))
    return (1 - (tp + eps) / (tp + alpha * fp + beta * fn + eps)).mean()


def focal_loss(pred, mask, alpha=0.75, gamma=2.0):
    r"""Focal BCE.  ``alpha`` weights the foreground class (0.75 favours fg)."""
    pred, mask = _split(pred, mask)
    bce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    prob = torch.sigmoid(pred)
    pt = prob * mask + (1 - prob) * (1 - mask)
    at = alpha * mask + (1 - alpha) * (1 - mask)
    return (at * (1 - pt) ** gamma * bce).mean()


def segmentation_loss(pred, mask, kind="tversky", pos_weight=None,
                      tversky_alpha=0.3, tversky_beta=0.7, focal_gamma=2.0):
    r"""Dispatcher used by ``IFCNet.compute_objective`` for every scale."""
    if kind == "structure":
        return structure_loss(pred, mask)
    if kind == "dice":
        return dice_loss(pred, mask) + bce_loss(pred, mask, pos_weight)
    if kind == "tversky":
        return tversky_loss(pred, mask, tversky_alpha, tversky_beta) + bce_loss(pred, mask, pos_weight)
    if kind == "focal":
        return focal_loss(pred, mask, gamma=focal_gamma) + dice_loss(pred, mask)
    if kind == "bce":
        return bce_loss(pred, mask, pos_weight)
    raise ValueError(f"Unknown loss kind {kind!r}; expected one of {LOSS_KINDS}")
