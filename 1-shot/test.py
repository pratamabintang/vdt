r""" Few-shot landslide segmentation testing (1-shot, RGB + DTM). """
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from model.mymodel import IFCNet
from model.losses import LOSS_KINDS
from common.logger import Logger
from common.evaluation import Evaluator
from common.metrics import BinaryMeter
from common import utils
from data.landslide import LandslideFSS


def unwrap(model):
    r"""Return the underlying module whether or not DataParallel wrapped it."""
    return model.module if isinstance(model, nn.DataParallel) else model


def resolve_device(name):
    r"""'auto' -> cuda when available, otherwise cpu."""
    if name == 'auto':
        name = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    return torch.device(name)


def save_visual(ori_rgb, q_mask, pred_mask, name, out_dir="./visual_maps"):
    """Saves query RGB + GT overlay + prediction overlay for inspection."""
    os.makedirs(out_dir, exist_ok=True)
    rgb = ori_rgb.cpu().squeeze().permute(1, 2, 0).numpy()
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    gt = q_mask.cpu().squeeze().numpy()
    pred = pred_mask.cpu().squeeze().numpy()

    def overlay(mask):
        out = rgb.copy()
        out[mask > 0] = [255, 50, 50]
        return out

    Image.fromarray(rgb).save(os.path.join(out_dir, name + '_query_rgb.png'))
    Image.fromarray(overlay(gt)).save(os.path.join(out_dir, name + '_query_GT.png'))
    Image.fromarray(overlay(pred)).save(os.path.join(out_dir, name + '_query_pred.png'))


def test(model, dataloader, loss_kind='tversky', pos_weight=None,
         amp=False, amp_dtype=torch.bfloat16, device=None):
    r"""Evaluate on the landslide test split."""
    meter = BinaryMeter()
    net = unwrap(model)
    if device is None:
        device = next(model.parameters()).device
    use_amp = amp and device.type == 'cuda'

    for query_rgb, query_dtm, query_mask, support_rgb, support_dtm, support_mask, subcls in dataloader:
        query_mask = query_mask.squeeze(1)
        query_rgb = query_rgb.to(device, non_blocking=True)
        query_dtm = query_dtm.to(device, non_blocking=True)
        support_rgb = support_rgb.to(device, non_blocking=True)
        support_dtm = support_dtm.to(device, non_blocking=True)
        support_mask = support_mask.squeeze(1).to(device, non_blocking=True)
        query_mask = query_mask.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logit_masks = model(query_rgb, query_dtm, support_rgb, support_dtm, support_mask)
            if not isinstance(logit_masks, list):
                logit_masks = [logit_masks]
            loss = net.compute_objective(logit_masks, query_mask,
                                         loss_kind=loss_kind, pos_weight=pos_weight)

        pred_mask = net.predict_mask(logit_masks[-1], query_mask.shape[1:])

        area_inter, area_union, area_pred, area_gt = Evaluator.classify_prediction(pred_mask, query_mask)
        meter.update(area_inter, area_union, subcls, loss, pred_b=area_pred, gt_b=area_gt)

    meter.write_result('Test', 0)
    miou, fb_iou, _ = meter.compute_iou()
    dice, _, _ = meter.compute_dice()
    precision, recall = meter.compute_precision_recall()
    return miou, fb_iou, dice, precision, recall, meter


def main(args, model):
    utils.fix_randseed(0)
    device = resolve_device(args.device)
    Logger.info('# device: %s | available GPUs: %d' % (device, torch.cuda.device_count()))

    model.to(device)
    if args.load == '':
        raise Exception('Pretrained model not specified.')
    model.load_state_dict(torch.load(os.path.join(args.load, "best_model.pt"), map_location=device))
    if device.type == 'cuda':
        model = nn.DataParallel(model)
    model.eval()

    Evaluator.initialize()

    use_amp = args.amp and device.type == 'cuda'
    amp_dtype = torch.bfloat16 if args.amp_dtype == 'bf16' else torch.float16
    if use_amp:
        Logger.info('AMP enabled: dtype=%s' % str(amp_dtype).replace('torch.', ''))
    Logger.info('Loss: %s (pos_weight=%s)' % (args.loss, args.pos_weight or 'none'))

    dataloader_test = LandslideFSS(
        data_dir=args.testdata,
        split='test',
        size=args.size,
        blacklist_path=args.blacklist,
        mode='test',
    )
    test_loader = torch.utils.data.DataLoader(
        dataloader_test, batch_size=1, shuffle=False,
        num_workers=args.nworker, pin_memory=True,
    )

    with torch.no_grad():
        test_miou, test_fb_iou, test_dice, prec, rec, meter = test(
            model, test_loader, loss_kind=args.loss, pos_weight=args.pos_weight or None,
            amp=use_amp, amp_dtype=amp_dtype, device=device)
    Logger.info('IoU(landslide): %5.2f \t FB-IoU: %5.2f \t Dice(landslide): %5.2f \t '
                'Prec(landslide): %5.2f \t Rec(landslide): %5.2f' %
                (test_miou.item(), test_fb_iou.item(), test_dice.item(), prec.item(), rec.item()))
    if meter.is_collapsed():
        Logger.info('[WARN] collapse: the model never predicts landslide (recall=0). '
                    'Check the training run (LR too low / epochs too few / loss).')
    Logger.info('==================== Finished Testing ====================')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Few-shot landslide segmentation test (1-shot, RGB + DTM)')
    parser.add_argument('--testdata', type=str, default='../datasets/landslide/test',
                        help='dir with test/{image,dtm,label} (or the test split dir itself)')
    parser.add_argument('--blacklist', type=str, default='../datasets/landslide/black_list.txt')
    parser.add_argument('--benchmark', type=str, default='fss', choices=['pascal', 'coco', 'fss'])
    parser.add_argument('--logpath', type=str, default='')
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--nworker', type=int, default=0)
    parser.add_argument('--load', type=str, default='../1-shot/saved_model/IFCNet/fold0')
    parser.add_argument('--backbone', type=str, default='resnet50',
                        choices=['vgg16', 'resnet50', 'resnet101',
                                 'sam2_hiera_t', 'sam2_hiera_s', 'sam2_hiera_b+', 'sam2_hiera_l'])
    parser.add_argument('--sam2-config', type=str, default='',
                        help="SAM2 hydra config (default: '<backbone>.yaml', e.g. sam2_hiera_l.yaml)")
    parser.add_argument('--sam2-ckpt', type=str, default=None,
                        help='SAM2 checkpoint (2.0), e.g. sam2_hiera_large.pt')
    parser.add_argument('--sam2-adapters', type=int, default=1, choices=[0, 1],
                        help='insert parameter-efficient adapters into the frozen SAM2 trunk')
    parser.add_argument('--model', type=str, default='IFCNet')
    parser.add_argument('--drop', type=float, default=0.2)
    parser.add_argument('--loss', type=str, default='tversky', choices=list(LOSS_KINDS),
                        help='loss used only to report Avg L (must match training for comparability)')
    parser.add_argument('--pos-weight', type=float, default=0.0,
                        help='BCE pos_weight for the rare foreground (0 = disabled)')
    parser.add_argument('--device', type=str, default='auto',
                        help="'auto' (cuda if available else cpu), 'cuda:0' or 'cpu'")
    parser.add_argument('--amp', action='store_true', help='enable automatic mixed precision (AMP)')
    parser.add_argument('--amp-dtype', type=str, default='bf16', choices=['bf16', 'fp16'],
                        help="AMP dtype: 'bf16' (recommended on L4/Ada) or 'fp16'")
    args = parser.parse_args()

    if args.model == 'IFCNet':
        Logger.initialize(args, training=True, modelname="IFCNet")
        # use_original_imgsize=True -> native logits (upsampled only for metrics)
        model = IFCNet(args.backbone, True, dropout=args.drop,
                       sam2_config=args.sam2_config or None,
                       sam2_ckpt=args.sam2_ckpt,
                       sam2_adapters=bool(args.sam2_adapters))
    else:
        raise KeyError(f"model version {args.model} not defined")

    main(args, model)
