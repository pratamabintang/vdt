r""" Loss functions for few-shot landslide segmentation. """
import torch
import torch.nn.functional as F


def structure_loss(pred, mask):
    r"""Weighted BCE + weighted IoU loss (structure-aware).

    ``pred`` may be either the 2-channel classifier output ``(B,2,H,W)`` or the
    single foreground logit map ``(B,1,H,W)``; ``mask`` may be ``(B,H,W)`` or
    ``(B,1,H,W)`` with values in {0,1}.
    """
    if pred.dim() == 4 and pred.size(1) == 2:
        pred = pred[:, 1:2]                # foreground logits only
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    mask = mask.float()

    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()
