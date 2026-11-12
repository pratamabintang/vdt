r""" Hypercorrelation Squeeze Network """
import contextlib
from functools import reduce
from operator import add

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet
from torchvision.models import vgg

from .Trifuse import *
from .base.feature import extract_feat_vgg, extract_feat_res, extract_feat_sam2
from .base.correlation import Correlation
from .losses import segmentation_loss

import numpy as np


class IFCNet(nn.Module):
    def __init__(self, backbone, use_original_imgsize, dropout=0.0, vis = False,
                 sam2_config=None, sam2_ckpt=None, sam2_adapters=True):
        super().__init__()
        # add supervision
        self.backbone_type = backbone
        # True  -> forward returns logits at their native resolution (3 scales,
        #          stride 8/16/32); the loss then runs per-scale and the caller
        #          upsamples only the final map for the metrics.  (train/test use
        #          this.)
        # False -> legacy: every scale is bilinearly upsampled to the input size
        #          inside forward.
        self.use_original_imgsize = use_original_imgsize
        # resnet/vgg are fully frozen and wrapped in torch.no_grad(); SAM2 keeps a
        # gradient path through its adapters (trunk params are frozen).
        self.frozen_grad_backbone = True
        if backbone == 'vgg16':
            self.backbone = vgg.vgg16(pretrained=True)
            self.feat_ids = [17, 19, 21, 24, 26, 28, 30]
            self.extract_feats = extract_feat_vgg
            self.nbottlenecks = [2, 2, 3, 3, 3, 1]
            self.multilevel_ch = [256,512,512] # l4,l3,l2
        elif backbone == 'resnet50':
            self.backbone = resnet.resnet50(pretrained=True)
            self.feat_ids = list(range(4, 17))
            self.extract_feats = extract_feat_res
            self.nbottlenecks = [3, 4, 6, 3]
            self.multilevel_ch = [512,1024,2048] # l4,l3,l2
        elif backbone == 'resnet101':
            self.backbone = resnet.resnet101(pretrained=True)
            self.feat_ids = list(range(4, 34))
            self.extract_feats = extract_feat_res
            self.nbottlenecks = [3, 4, 23, 3]
            self.multilevel_ch = [512,1024,2048] # l2,l3,l4
        elif backbone in ('sam2_hiera_t', 'sam2_hiera_s', 'sam2_hiera_b+', 'sam2_hiera_l'):
            from .sam2_encoder import SAM2Encoder
            cfg = sam2_config or (backbone + '.yaml')
            self.backbone = SAM2Encoder(cfg, sam2_ckpt, adapters=sam2_adapters)
            # one feature per level (x2, x3, x4) -> 3 levels fine->coarse
            self.feat_ids = [0, 1, 2]
            self.extract_feats = extract_feat_sam2
            self.nbottlenecks = [1, 1, 1, 1]
            self.multilevel_ch = list(self.backbone.out_channels)  # [x2,x3,x4]
            self.frozen_grad_backbone = not bool(sam2_adapters)
        else:
            raise Exception('Unavailable backbone: %s' % backbone)

        self.nlevel = len(self.multilevel_ch)
        self.vis = vis
        self.bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), self.nbottlenecks)))
        self.lids = reduce(add, [[i + 1] * x for i, x in enumerate(self.nbottlenecks)])
        # self.stack_ids = torch.tensor(self.lids).bincount().__reversed__().cumsum(dim=0)[:3]
        self.layer_ids = np.array([0]+self.nbottlenecks[1:]).cumsum()
        self.nbottlenecks = self.nbottlenecks[1:]
        self.backbone.eval() # backbone frozen

        # self.corr_mid = corr_mid*3 if len(corr_mid)==1 else corr_mid # correspond to l2,l3,l4

        self.multi_smi_fuse = nn.ModuleList([ConvModule(self.nbottlenecks[i]*2,self.nbottlenecks[i],1) 
                                             for i in range(len(self.nbottlenecks))]) # correspond to l2,l3,l4
        

        self.tri_mid_ch=128
        self.tri_out_ch=64
        self.TriFusel4 = BiCorr_v1(in_ch=self.multilevel_ch[-1],mid_ch=self.tri_mid_ch,out_ch=self.tri_out_ch,dropout=dropout)
        self.TriFusel3 = BiCorr_v1(in_ch=self.multilevel_ch[-2],mid_ch=self.tri_mid_ch,out_ch=self.tri_out_ch,dropout=dropout)
        #self.TriFusel2 = TriCorr_v2_3(in_ch=self.multilevel_ch[-1],mid_ch=self.tri_mid_ch,out_ch=self.tri_out_ch,dropout=0.2)

        self.Fusion = ConvModule(self.tri_out_ch+self.nbottlenecks[-1],self.tri_mid_ch,1)
        self.deblock1 = ConvModule(self.tri_out_ch+self.tri_mid_ch+self.nbottlenecks[-2],self.tri_out_ch,1)
        self.deblock2 = ConvModule(self.tri_out_ch+self.nbottlenecks[-3],self.tri_out_ch,1)

        self.side3 = ConvModule(self.tri_mid_ch,2,3,mode="conv")   # raw logits (no BN/ReLU)
        self.side2 = ConvModule(self.tri_out_ch,2,3,mode="conv")
        self.side1 = ConvModule(self.tri_out_ch,2,3,mode="conv")
        #self.side0 = ConvModule(2*3,2,1,mode="sigmoid")

        

    def _upsample_like(self, src, tar):
        src = F.interpolate(src, size=tar.shape[2:], mode='bilinear', align_corners=True)

        return src

    def forward(self, query_img, query_img_th, support_img, support_img_th, support_mask):
        feat_ctx = torch.no_grad() if self.frozen_grad_backbone else contextlib.nullcontext()
        with feat_ctx:
            # only the output of last layer in each level is used
            query_feats = self.extract_feats(query_img, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
            support_feats = self.extract_feats(support_img, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
            support_feats = self.mask_feature(support_feats, support_mask.clone())

            query_feats_th = self.extract_feats(query_img_th, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
            support_feats_th = self.extract_feats(support_img_th, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
            support_feats_th = self.mask_feature(support_feats_th, support_mask.clone())        

         # pick l2, l3, l4
        query_feats = [query_feats[self.layer_ids[i]:self.layer_ids[i+1]] for i in range(0,len(self.layer_ids)-1)]
        support_feats = [support_feats[self.layer_ids[i]:self.layer_ids[i+1]] for i in range(0,len(self.layer_ids)-1)]
        query_feats_th = [query_feats_th[self.layer_ids[i]:self.layer_ids[i+1]] for i in range(0,len(self.layer_ids)-1)]
        support_feats_th = [support_feats_th[self.layer_ids[i]:self.layer_ids[i+1]] for i in range(0,len(self.layer_ids)-1)]
        
        multi_sim = []
        if self.vis:
            vsim_maps = []
            tsim_maps = []
        for idx,(qv_layer,sv_layer,qt_layer,st_layer) in enumerate(zip(query_feats,support_feats,query_feats_th,support_feats_th)):
            # 3 levels in total
            vsim = []
            tsim = []
            for iidx, (qv,sv,qt,st) in enumerate(zip(qv_layer,sv_layer,qt_layer,st_layer)):             
                vsim.append(Correlation.multi_similarity(qv,sv))
                tsim.append(Correlation.multi_similarity(qt,st))
                
            trisim = self.multi_smi_fuse[idx](torch.cat(vsim+tsim,dim=1))
            multi_sim.append(trisim)
            if self.vis:
                vsim_maps.append(vsim)
                tsim_maps.append(tsim)
            
        
        # multi_sim ->[l2,l3,l4] each level is (bsz,1,qH,qW)
             
        ############## RGB is assigned as main modality, thermal as aux #############
        Trimap_l4 = self.TriFusel4([query_feats[-1][-1],query_feats_th[-1][-1]],
                                 [support_feats[-1][-1],support_feats_th[-1][-1]]) # features = [fV,fT]
        Trimap_l4 = torch.cat(Trimap_l4,dim=1)
        if(self.vis):vis_l4 = [query_feats[-1][-1],query_feats_th[-1][-1],Trimap_l4]


        Trimap_l3 = self.TriFusel3([query_feats[-2][-1],query_feats_th[-2][-1]],
                                 [support_feats[-2][-1],support_feats_th[-2][-1]])
        Trimap_l3 = torch.cat(Trimap_l3,dim=1)
        if(self.vis):vis_l3 = [query_feats[-2][-1],query_feats_th[-2][-1],Trimap_l3]

        logit_masks=[]

        out3 = self.Fusion(torch.cat([Trimap_l4,multi_sim[-1]],dim=1))
        logit_masks.append(self.side3(out3))

        out2 = self.deblock1(torch.cat([Trimap_l3,self._upsample_like(out3,multi_sim[-2]),multi_sim[-2]],dim=1))
        logit_masks.append(self.side2(out2))

        out1 = self.deblock2(torch.cat([self._upsample_like(out2,multi_sim[-3]),multi_sim[-3]],dim=1))
        logit_masks.append(self.side1(out1))

        #logit_masks.append(self.side0(torch.cat([out1,out2,out3],dim=1)))
        

        if not self.use_original_imgsize:
            logit_masks=[F.interpolate(logit_mask, support_img.size()[2:], mode='bilinear', align_corners=True) for logit_mask in logit_masks]
                
        # decoder output 2 channel: indicate probabilities of foreground and background
        if self.vis:
            return logit_masks,vsim_maps,tsim_maps,multi_sim,vis_l3,vis_l4
        else:
            return logit_masks

    def predict_mask(self, logit_mask, size):
        r"""argmax over the 2 logit channels, upsampled to ``size`` (H, W)."""
        size = tuple(size)
        if logit_mask.shape[2:] != size:
            logit_mask = F.interpolate(logit_mask, size=size, mode='bilinear', align_corners=True)
        return logit_mask.float().argmax(dim=1)

    def mask_feature(self, features, support_mask):
        for idx, feature in enumerate(features):
            mask = F.interpolate(support_mask.unsqueeze(1).float(), feature.size()[2:], mode='bilinear', align_corners=True)
            features[idx] = features[idx] * mask
        return features

    def compute_objective(self, logit_masks, gt_mask, auxloss_weight=0.4,
                          loss_kind='tversky', pos_weight=None):
        r"""Deep-supervision loss.

        Every scale is scored at its **native** resolution (the ground truth is
        down-sampled to the logit size) so the coarse 16x16 / 32x32 maps are not
        diluted by the 98% background of a full 512x512 grid.

        Deep supervision: finest scale weight 1.0, auxiliary scales weight
        ``auxloss_weight``.  ``loss_kind`` selects the foreground-aware region
        loss (see ``model.losses``): ``tversky`` (default), ``dice``, ``focal``,
        ``bce`` or the legacy ``structure``.
        """
        if not isinstance(logit_masks, list):
            logit_masks = [logit_masks]

        gt = gt_mask.unsqueeze(1) if gt_mask.dim() == 3 else gt_mask
        gt = gt.float()

        last = len(logit_masks) - 1
        loss = 0
        for i, logit_mask in enumerate(logit_masks):
            w = 1.0 if i == last else auxloss_weight
            scale_gt = gt
            if scale_gt.shape[2:] != logit_mask.shape[2:]:
                scale_gt = F.interpolate(scale_gt, size=logit_mask.shape[2:], mode='nearest')
            loss = loss + w * segmentation_loss(logit_mask, scale_gt, kind=loss_kind,
                                                pos_weight=pos_weight)
        return loss

    def train_mode(self):
        self.train()
        self.backbone.eval()  # to prevent BN from learning data statistics with exponential averaging
