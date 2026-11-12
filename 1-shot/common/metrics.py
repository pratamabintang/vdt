r""" Binary (landslide) evaluation meter and logging helpers. """
import torch

from common.logger import Logger


class BinaryMeter:
    r"""Accumulates intersection/union for a single foreground class.

    Compatible with ``Evaluator.classify_prediction`` outputs (shape ``[2, B]``,
    row 0 = background, row 1 = foreground).
    """

    def __init__(self):
        self.inter = torch.zeros(2, dtype=torch.float64)
        self.union = torch.zeros(2, dtype=torch.float64)
        self.loss_buf = []

    def update(self, inter_b, union_b, class_id=None, loss=None):
        inter_b = inter_b.detach().double().cpu()
        union_b = union_b.detach().double().cpu()
        if inter_b.dim() == 2:
            inter_b = inter_b.sum(dim=1)
            union_b = union_b.sum(dim=1)
        self.inter += inter_b
        self.union += union_b
        if loss is not None:
            self.loss_buf.append(float(loss.detach().cpu()))

    def compute_iou(self):
        iou = self.inter / self.union.clamp(min=1.0)
        iou_landslide = iou[1] * 100.0
        fb_iou = (self.inter.sum() / self.union.sum().clamp(min=1.0)) * 100.0
        return iou_landslide, fb_iou, iou

    def compute_dice(self):
        r"""Dice/F1 per class. pred + gt = union + inter."""
        denom = (self.union + self.inter).clamp(min=1.0)
        dice = 2.0 * self.inter / denom
        dice_landslide = dice[1] * 100.0
        mean_dice = dice.mean() * 100.0
        return dice_landslide, mean_dice, dice

    def _avg_loss(self):
        return sum(self.loss_buf) / len(self.loss_buf) if self.loss_buf else 0.0

    def write_result(self, split, epoch):
        iou, fb_iou, per_class = self.compute_iou()
        dice_fg, dice_mean, per_dice = self.compute_dice()
        msg = '\n*** %s ' % split
        msg += '[@Epoch %02d] ' % epoch
        msg += 'Avg L: %6.5f  ' % self._avg_loss()
        msg += 'IoU(landslide): %5.2f   ' % iou
        msg += 'IoU(bg): %5.2f   ' % (per_class[0] * 100.0)
        msg += 'FB-IoU: %5.2f   ' % fb_iou
        msg += 'Dice(landslide): %5.2f   ' % dice_fg
        msg += 'Dice(bg): %5.2f   ' % (per_dice[0] * 100.0)
        msg += '***\n'
        Logger.info(msg)
