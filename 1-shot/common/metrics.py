r""" Binary (landslide) evaluation meter and logging helpers. """
import torch

from common.logger import Logger


class BinaryMeter:
    r"""Accumulates intersection/union for a single foreground class.

    Compatible with ``Evaluator.classify_prediction`` outputs (shape ``[2, B]``,
    row 0 = background, row 1 = foreground).  Predicted / ground-truth class areas
    are tracked as well so precision and recall of the (rare) landslide class can
    be reported — a collapsed all-background model shows up as recall 0.
    """

    def __init__(self):
        self.inter = torch.zeros(2, dtype=torch.float64)
        self.union = torch.zeros(2, dtype=torch.float64)
        self.pred = torch.zeros(2, dtype=torch.float64)
        self.gt = torch.zeros(2, dtype=torch.float64)
        self.loss_buf = []

    @staticmethod
    def _sum2(vec):
        if vec is None:
            return None
        vec = vec.detach().double().cpu()
        return vec.sum(dim=1) if vec.dim() == 2 else vec

    def update(self, inter_b, union_b, class_id=None, loss=None, pred_b=None, gt_b=None):
        self.inter += self._sum2(inter_b)
        self.union += self._sum2(union_b)
        pred_b = self._sum2(pred_b)
        gt_b = self._sum2(gt_b)
        if pred_b is not None:
            self.pred += pred_b
        if gt_b is not None:
            self.gt += gt_b
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

    def compute_precision_recall(self):
        r"""Foreground precision/recall (``%``).  Recall == 0 => model never
        predicts landslide, i.e. it has collapsed to all-background."""
        tp = self.inter[1]
        precision = (tp / self.pred[1].clamp(min=1.0)) * 100.0
        recall = (tp / self.gt[1].clamp(min=1.0)) * 100.0
        return precision, recall

    def foreground_iou(self):
        return (self.inter[1] / self.union[1].clamp(min=1.0)) * 100.0

    def is_collapsed(self):
        r"""True when nothing was ever predicted as landslide."""
        return float(self.inter[1]) == 0.0

    def _avg_loss(self):
        return sum(self.loss_buf) / len(self.loss_buf) if self.loss_buf else 0.0

    def write_result(self, split, epoch):
        iou, fb_iou, per_class = self.compute_iou()
        dice_fg, dice_mean, per_dice = self.compute_dice()
        precision, recall = self.compute_precision_recall()
        msg = '\n*** %s ' % split
        msg += '[@Epoch %02d] ' % epoch
        msg += 'Avg L: %6.5f  ' % self._avg_loss()
        msg += 'IoU(landslide): %5.2f   ' % iou
        msg += 'IoU(bg): %5.2f   ' % (per_class[0] * 100.0)
        msg += 'FB-IoU: %5.2f   ' % fb_iou
        msg += 'Dice(landslide): %5.2f   ' % dice_fg
        msg += 'Dice(bg): %5.2f   ' % (per_dice[0] * 100.0)
        msg += 'Prec(landslide): %5.2f   ' % precision
        msg += 'Rec(landslide): %5.2f   ' % recall
        msg += '***\n'
        Logger.info(msg)
