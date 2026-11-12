r""" Few-shot landslide segmentation training (1-shot, RGB + DTM). """
import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim

from model.mymodel import IFCNet
from model.losses import LOSS_KINDS
from common.logger import Logger
from common.evaluation import Evaluator
from common.metrics import BinaryMeter
from common import utils
from data.landslide import LandslideFSS


def resolve_device(name):
    r"""'auto' -> cuda when available, otherwise cpu (lets you smoke-test on CPU)."""
    if name == 'auto':
        name = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    return torch.device(name)


def unwrap(model):
    r"""Return the underlying module whether or not DataParallel wrapped it."""
    return model.module if isinstance(model, nn.DataParallel) else model


def build_amp(args, device):
    r"""Resolve AMP settings from CLI args. bf16 needs no GradScaler; fp16 does."""
    use_amp = bool(args.amp) and device.type == 'cuda'
    dtype = torch.bfloat16 if args.amp_dtype == 'bf16' else torch.float16
    scaler = None
    if use_amp and dtype == torch.float16:
        try:
            scaler = torch.amp.GradScaler('cuda')
        except (AttributeError, TypeError):
            scaler = torch.cuda.amp.GradScaler()
    return use_amp, dtype, scaler


def train(epoch, model, dataloader, optimizer, training, auxloss_weight=0.4,
          loss_kind='tversky', pos_weight=None, amp=False, amp_dtype=torch.bfloat16,
          scaler=None, device=None):
    r"""One epoch of training / validation over few-shot landslide episodes."""
    # Force randomness during training / freeze randomness during validation
    utils.fix_randseed(None) if training else utils.fix_randseed(0)
    net = unwrap(model)
    net.train_mode() if training else net.eval()
    meter = BinaryMeter()

    if device is None:
        device = next(model.parameters()).device
    use_amp = amp and device.type == 'cuda'

    for idx, (input_rgb, input_dtm, target, s_rgb, s_dtm, s_mask, subcls) in enumerate(dataloader):
        # 1. Move the episode to the device (RGB = anchor, DTM = aux modality)
        target = target.squeeze(1)
        input_rgb = input_rgb.to(device, non_blocking=True)
        input_dtm = input_dtm.to(device, non_blocking=True)
        s_rgb = s_rgb.to(device, non_blocking=True)
        s_dtm = s_dtm.to(device, non_blocking=True)
        s_mask = s_mask.squeeze(1).to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # 2. Forward pass + loss under AMP autocast (bf16 on L4)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logit_masks = model(input_rgb, input_dtm, s_rgb, s_dtm, s_mask)
            loss = net.compute_objective(logit_masks, target, auxloss_weight,
                                         loss_kind=loss_kind, pos_weight=pos_weight)

        # logits are at native resolution -> upsample only the finest for metrics
        if not isinstance(logit_masks, list):
            logit_masks = [logit_masks]
        pred_mask = net.predict_mask(logit_masks[-1], target.shape[1:])

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
        area_inter, area_union, area_pred, area_gt = Evaluator.classify_prediction(pred_mask, target)
        meter.update(area_inter, area_union, subcls, loss,
                     pred_b=area_pred, gt_b=area_gt)

    meter.write_result('Training' if training else 'Validation', epoch)
    avg_loss = meter._avg_loss()
    miou, fb_iou, _ = meter.compute_iou()
    dice, _, _ = meter.compute_dice()
    _, recall = meter.compute_precision_recall()

    if not training and meter.is_collapsed():
        Logger.info('[WARN] collapse: validation never predicts landslide '
                    '(recall(landslide)=0.00). Check LR / loss / epochs.')

    return avg_loss, miou, fb_iou, dice, recall


def build_loader(datadir, split, size, bsz, nworker, blacklist, episodes=0):
    is_train = split == 'train'
    if is_train and episodes and episodes < bsz:
        raise ValueError(
            f"--episodes={episodes} < --bsz={bsz}: an epoch would contain 0 batches and "
            f"the model would not train at all. Use --episodes >= --bsz (or 0 = all positives)."
        )
    dataset = LandslideFSS(
        data_dir=datadir,
        split=split,
        size=size,
        blacklist_path=blacklist,
        mode='train' if is_train else 'val',
        episodes_per_epoch=episodes,
    )
    # drop_last only when it cannot empty the loader (short episodes must train)
    drop_last = is_train and len(dataset) >= bsz
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=bsz,
        shuffle=is_train,
        num_workers=nworker,
        pin_memory=True,
        drop_last=drop_last,
    )
    if is_train:
        dropped = len(dataset) - len(loader) * bsz
        if dropped:
            Logger.info('# train: %d episodes -> %d batches (drop_last=%s, %d episodes dropped)'
                        % (len(dataset), len(loader), drop_last, dropped))
    return loader


def main(args, model):
    device = resolve_device(args.device)
    Logger.info('# device: %s | available GPUs: %d' % (device, torch.cuda.device_count()))

    model = nn.DataParallel(model) if device.type == 'cuda' else model
    model.to(device)

    optimizer = optim.Adam([{"params": model.parameters(), "lr": args.lr}])
    Evaluator.initialize()
    start_epoch = 0

    use_amp, amp_dtype, scaler = build_amp(args, device)
    if use_amp:
        Logger.info('AMP enabled: dtype=%s, grad_scaler=%s' %
                    (str(amp_dtype).replace('torch.', ''), scaler is not None))
    Logger.info('Loss: %s (auxloss=%s, pos_weight=%s)' %
                (args.loss, args.auxloss, args.pos_weight or 'none'))

    if args.resume is not None:
        model.load_state_dict(torch.load(os.path.join(args.resume, 'best_model.pt'), map_location=device))
        ckp = torch.load(os.path.join(args.resume, 'best_ckp.pt'), map_location=device)
        optimizer.load_state_dict(ckp['optimizer'])
        start_epoch = ckp['epoch'] + 1

    train_loader = build_loader(args.traindata, 'train', args.size, args.bsz, args.nworker, args.blacklist, args.episodes)
    val_loader = build_loader(args.valdata, 'val', args.size, args.bsz, args.nworker, args.blacklist)
    Logger.info('# train episodes/epoch: %d (%d batches) | val episodes: %d' %
                (len(train_loader.dataset), len(train_loader), len(val_loader.dataset)))
    if len(train_loader) == 0:
        raise RuntimeError('Train loader has 0 batches -> nothing would be trained.')

    best_val_miou = float('-inf')
    best_val_fb_miou = float('-inf')
    best_epoch = -1
    for epoch in range(start_epoch, args.epoch):
        trn_loss, trn_miou, trn_fb_iou, trn_dice, trn_rec = train(
            epoch, model, train_loader, optimizer, training=True,
            auxloss_weight=args.auxloss, loss_kind=args.loss, pos_weight=args.pos_weight or None,
            amp=use_amp, amp_dtype=amp_dtype, scaler=scaler, device=device)
        val_loss, val_miou, val_fb_iou, val_dice, val_rec = train(
            epoch, model, val_loader, optimizer, training=False,
            auxloss_weight=args.auxloss, loss_kind=args.loss, pos_weight=args.pos_weight or None,
            amp=use_amp, amp_dtype=amp_dtype, scaler=None, device=device)

        if val_miou > best_val_miou:
            best_val_miou = val_miou
            best_val_fb_miou = val_fb_iou
            best_epoch = epoch
            Logger.save_model_miou(model, best_epoch, val_miou)
        Logger.save_ckp(optimizer, epoch)

        print("%d epoch ,%.2f is the best IoU(landslide), %.2f is the best FB-IoU "
              "(val recall=%.2f)" %
              (best_epoch, best_val_miou, best_val_fb_miou, val_rec))

        if best_val_miou <= 0.0:
            Logger.info('[WARN] best IoU(landslide) is still %.2f after epoch %d: the model '
                        'predicts background only. Raise --lr (>=3e-4), raise --epochs, or use '
                        '--loss tversky/dice/focal.' % (best_val_miou, epoch))

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
                        help='query episodes per training epoch (0 = one pass over all positive tiles); '
                             'must be 0 or >= bsz')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='learning rate; with a frozen backbone >=3e-4 is recommended')
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
    parser.add_argument('--loss', type=str, default='tversky', choices=list(LOSS_KINDS),
                        help='segmentation loss: tversky (default), dice, focal, bce or legacy structure')
    parser.add_argument('--pos-weight', type=float, default=0.0,
                        help='BCE pos_weight for the rare foreground (0 = disabled); ~5-20 helps at 2%% fg')
    parser.add_argument('--device', type=str, default='auto',
                        help="'auto' (cuda if available else cpu), 'cuda:0' or 'cpu'")
    parser.add_argument('--amp', action='store_true', help='enable automatic mixed precision (AMP)')
    parser.add_argument('--amp-dtype', type=str, default='bf16', choices=['bf16', 'fp16'],
                        help="AMP dtype: 'bf16' (recommended on L4/Ada) or 'fp16' (uses GradScaler)")
    args = parser.parse_args()

    if args.model == 'IFCNet':
        Logger.initialize(args, training=True, modelname="IFCNet")
        # use_original_imgsize=True -> keep native logits (loss runs per scale)
        model = IFCNet(args.backbone, True, dropout=args.drop,
                       sam2_config=args.sam2_config or None,
                       sam2_ckpt=args.sam2_ckpt,
                       sam2_adapters=bool(args.sam2_adapters))
    else:
        raise KeyError(f"model version {args.model} not defined")

    Logger.log_params(model)
    main(args, model)
