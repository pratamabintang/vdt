r""" Few-shot landslide segmentation training (1-shot, RGB + DTM). """
import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim

from model.mymodel import IFCNet
from common.logger import Logger
from common.evaluation import Evaluator
from common.metrics import BinaryMeter
from common import utils
from data.landslide import LandslideFSS


def build_amp(args):
    r"""Resolve AMP settings from CLI args. bf16 needs no GradScaler; fp16 does."""
    use_amp = args.amp and torch.cuda.is_available()
    dtype = torch.bfloat16 if args.amp_dtype == 'bf16' else torch.float16
    scaler = None
    if use_amp and dtype == torch.float16:
        try:
            scaler = torch.amp.GradScaler('cuda')
        except (AttributeError, TypeError):
            scaler = torch.cuda.amp.GradScaler()
    return use_amp, dtype, scaler


def train(epoch, model, dataloader, optimizer, training, auxloss_weight=0.4,
          amp=False, amp_dtype=torch.bfloat16, scaler=None):
    r"""One epoch of training / validation over few-shot landslide episodes."""
    # Force randomness during training / freeze randomness during validation
    utils.fix_randseed(None) if training else utils.fix_randseed(0)
    model.module.train_mode() if training else model.module.eval()
    meter = BinaryMeter()

    use_amp = amp and torch.cuda.is_available()
    for idx, (input_rgb, input_dtm, target, s_rgb, s_dtm, s_mask, subcls) in enumerate(dataloader):
        # 1. Forward pass (RGB = anchor, DTM = aux modality)
        s_mask = s_mask.squeeze(1)
        target = target.squeeze(1)
        input_rgb = input_rgb.cuda(non_blocking=True)
        input_dtm = input_dtm.cuda(non_blocking=True)
        s_rgb = s_rgb.cuda(non_blocking=True)
        s_dtm = s_dtm.cuda(non_blocking=True)
        s_mask = s_mask.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # 2. Forward pass + loss under AMP autocast (bf16 on L4)
        with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
            logit_masks = model(input_rgb, input_dtm, s_rgb, s_dtm, s_mask)
            if not isinstance(logit_masks, list):
                logit_masks = [logit_masks]
            loss = model.module.compute_objective(logit_masks, target, auxloss_weight)

        pred_mask = logit_masks[-1].float().argmax(dim=1)

        # 3. Update model parameters (GradScaler only needed for fp16)
        if training:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        # 4. Evaluate prediction
        area_inter, area_union = Evaluator.classify_prediction(pred_mask, target)
        meter.update(area_inter, area_union, subcls, loss)

    meter.write_result('Training' if training else 'Validation', epoch)
    avg_loss = meter._avg_loss()
    miou, fb_iou, _ = meter.compute_iou()
    dice, _, _ = meter.compute_dice()

    return avg_loss, miou, fb_iou, dice


def build_loader(datadir, split, size, bsz, nworker, blacklist, episodes=0):
    dataset = LandslideFSS(
        data_dir=datadir,
        split=split,
        size=size,
        blacklist_path=blacklist,
        mode='train' if split == 'train' else 'val',
        episodes_per_epoch=episodes,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=bsz,
        shuffle=(split == 'train'),
        num_workers=nworker,
        pin_memory=True,
        drop_last=(split == 'train'),
    )
    return loader


def main(args, model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Logger.info('# available GPUs: %d' % torch.cuda.device_count())

    model = nn.DataParallel(model)
    model.to(device)

    optimizer = optim.Adam([{"params": model.parameters(), "lr": args.lr}])
    Evaluator.initialize()
    start_epoch = 0

    use_amp, amp_dtype, scaler = build_amp(args)
    if use_amp:
        Logger.info('AMP enabled: dtype=%s, grad_scaler=%s' %
                    (str(amp_dtype).replace('torch.', ''), scaler is not None))

    if args.resume is not None:
        model.load_state_dict(torch.load(os.path.join(args.resume, 'best_model.pt'), map_location=device))
        ckp = torch.load(os.path.join(args.resume, 'best_ckp.pt'), map_location=device)
        optimizer.load_state_dict(ckp['optimizer'])
        start_epoch = ckp['epoch'] + 1

    train_loader = build_loader(args.traindata, 'train', args.size, args.bsz, args.nworker, args.blacklist, args.episodes)
    val_loader = build_loader(args.valdata, 'val', args.size, args.bsz, args.nworker, args.blacklist)
    Logger.info('# train episodes/epoch: %d (%d batches) | val episodes: %d' %
                (len(train_loader.dataset), len(train_loader), len(val_loader.dataset)))

    best_val_miou = float('-inf')
    best_val_fb_miou = float('-inf')
    best_epoch = -1
    for epoch in range(start_epoch, args.epoch):
        trn_loss, trn_miou, trn_fb_iou, trn_dice = train(
            epoch, model, train_loader, optimizer, training=True,
            auxloss_weight=args.auxloss, amp=use_amp, amp_dtype=amp_dtype, scaler=scaler)
        with torch.no_grad():
            val_loss, val_miou, val_fb_iou, val_dice = train(
                epoch, model, val_loader, optimizer, training=False,
                auxloss_weight=args.auxloss, amp=use_amp, amp_dtype=amp_dtype, scaler=None)

        if val_miou > best_val_miou:
            best_val_miou = val_miou
            best_val_fb_miou = val_fb_iou
            best_epoch = epoch
            Logger.save_model_miou(model, best_epoch, val_miou)
        Logger.save_ckp(optimizer, epoch)

        print("%d epoch ,%.2f is the best IoU(landslide), %.2f is the best FB-IoU" %
              (best_epoch, best_val_miou, best_val_fb_miou))

        Logger.tbd_writer.add_scalars('data/loss', {'trn_loss': trn_loss, 'val_loss': val_loss}, epoch)
        Logger.tbd_writer.add_scalars('data/miou', {'trn_miou': trn_miou, 'val_miou': val_miou}, epoch)
        Logger.tbd_writer.add_scalars('data/fb_iou', {'trn_fb_iou': trn_fb_iou, 'val_fb_iou': val_fb_iou}, epoch)
        Logger.tbd_writer.add_scalars('data/dice', {'trn_dice': trn_dice, 'val_dice': val_dice}, epoch)
        Logger.tbd_writer.flush()
    Logger.tbd_writer.close()
    Logger.info('==================== Finished Training ====================')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Few-shot landslide segmentation (1-shot, RGB + DTM)')
    parser.add_argument('--traindata', type=str, default='../datasets/landslide/train',
                        help='dir with train/{image,dtm,label} (or the train split dir itself)')
    parser.add_argument('--valdata', type=str, default='../datasets/landslide/val',
                        help='dir with val/{image,dtm,label} (or the val split dir itself)')
    parser.add_argument('--blacklist', type=str, default='../datasets/landslide/black_list.txt',
                        help='single blacklist covering broken tiles in train/val/test')
    parser.add_argument('--benchmark', type=str, default='fss', choices=['pascal', 'coco', 'fss'])
    parser.add_argument('--logpath', type=str, default='')
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--bsz', type=int, default=2)
    parser.add_argument('--episodes', type=int, default=0,
                        help='query episodes per training epoch (0 = one pass over all positive tiles)')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--nworker', type=int, default=2)
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
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--drop', type=float, default=0.2)
    parser.add_argument('--auxloss', type=float, default=0.4,
                        help='weight of auxiliary (coarse) scale losses in deep supervision')
    parser.add_argument('--amp', action='store_true', help='enable automatic mixed precision (AMP)')
    parser.add_argument('--amp-dtype', type=str, default='bf16', choices=['bf16', 'fp16'],
                        help="AMP dtype: 'bf16' (recommended on L4/Ada) or 'fp16' (uses GradScaler)")
    args = parser.parse_args()

    if args.model == 'IFCNet':
        Logger.initialize(args, training=True, modelname="IFCNet")
        model = IFCNet(args.backbone, False, dropout=args.drop,
                       sam2_config=args.sam2_config or None,
                       sam2_ckpt=args.sam2_ckpt,
                       sam2_adapters=bool(args.sam2_adapters))
    else:
        raise KeyError(f"model version {args.model} not defined")

    Logger.log_params(model)
    main(args, model)
