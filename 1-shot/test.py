r""" Few-shot landslide segmentation testing (1-shot, RGB + DTM). """
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from model.mymodel import IFCNet
from common.logger import Logger
from common.evaluation import Evaluator
from common.metrics import BinaryMeter
from common import utils
from data.landslide import LandslideFSS


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


def test(model, dataloader, amp=False, amp_dtype=torch.bfloat16):
    r"""Evaluate on the landslide test split."""
    meter = BinaryMeter()
    use_amp = amp and torch.cuda.is_available()

    for query_rgb, query_dtm, query_mask, support_rgb, support_dtm, support_mask, subcls in dataloader:
        support_mask = support_mask.squeeze(1)
        query_mask = query_mask.squeeze(1)
        query_rgb = query_rgb.cuda(non_blocking=True)
        query_dtm = query_dtm.cuda(non_blocking=True)
        support_rgb = support_rgb.cuda(non_blocking=True)
        support_dtm = support_dtm.cuda(non_blocking=True)
        support_mask = support_mask.cuda(non_blocking=True)
        query_mask = query_mask.cuda(non_blocking=True)

        with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
            logit_masks = model(query_rgb, query_dtm, support_rgb, support_dtm, support_mask)
            if not isinstance(logit_masks, list):
                logit_masks = [logit_masks]
            loss = model.module.compute_objective(logit_masks, query_mask)

        pred_mask = logit_masks[-1].float().argmax(dim=1)

        area_inter, area_union = Evaluator.classify_prediction(pred_mask, query_mask)
        meter.update(area_inter, area_union, subcls, loss)

    meter.write_result('Test', 0)
    miou, fb_iou, _ = meter.compute_iou()
    dice, _, _ = meter.compute_dice()
    return miou, fb_iou, dice


def main(args, model):
    utils.fix_randseed(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Logger.info('# available GPUs: %d' % torch.cuda.device_count())

    model.to(device)
    if args.load == '':
        raise Exception('Pretrained model not specified.')
    model.load_state_dict(torch.load(os.path.join(args.load, "best_model.pt"), map_location=device))
    model = nn.DataParallel(model)
    model.eval()

    Evaluator.initialize()

    use_amp = args.amp and torch.cuda.is_available()
    amp_dtype = torch.bfloat16 if args.amp_dtype == 'bf16' else torch.float16
    if use_amp:
        Logger.info('AMP enabled: dtype=%s' % str(amp_dtype).replace('torch.', ''))

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
        test_miou, test_fb_iou, test_dice = test(model, test_loader, amp=use_amp, amp_dtype=amp_dtype)
    Logger.info('IoU(landslide): %5.2f \t FB-IoU: %5.2f \t Dice(landslide): %5.2f' %
                (test_miou.item(), test_fb_iou.item(), test_dice.item()))
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
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['vgg16', 'resnet50', 'resnet101'])
    parser.add_argument('--model', type=str, default='IFCNet')
    parser.add_argument('--drop', type=float, default=0.2)
    parser.add_argument('--amp', action='store_true', help='enable automatic mixed precision (AMP)')
    parser.add_argument('--amp-dtype', type=str, default='bf16', choices=['bf16', 'fp16'],
                        help="AMP dtype: 'bf16' (recommended on L4/Ada) or 'fp16'")
    args = parser.parse_args()

    if args.model == 'IFCNet':
        Logger.initialize(args, training=True, modelname="IFCNet")
        model = IFCNet(args.backbone, False, dropout=args.drop)
    else:
        raise KeyError(f"model version {args.model} not defined")

    main(args, model)
