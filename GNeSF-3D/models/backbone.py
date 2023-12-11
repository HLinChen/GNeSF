'''
Author: chusan.chl
Date: 2021-12-04 05:07:44
LastEditors: chusan.chl
LastEditTime: 2022-05-21 21:19:00
Description: 
FilePath: /NeuralRecon/models/backbone.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from loguru import logger

from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model as det2_build_model
from detectron2.projects.deeplab import add_deeplab_config

from seg2d.mask2former import add_maskformer2_config
from ops.comm import is_main_process


def _round_to_multiple_of(val, divisor, round_up_bias=0.9):
    """ Asymmetric rounding to make `val` divisible by `divisor`. With default
    bias, will round up, unless the number is no more than 10% greater than the
    smaller divisible value, i.e. (83, 8) -> 80, but (84, 8) -> 88. """
    assert 0.0 < round_up_bias < 1.0
    new_val = max(divisor, int(val + divisor / 2) // divisor * divisor)
    return new_val if new_val >= round_up_bias * val else new_val + divisor


def _get_depths(alpha):
    """ Scales tensor depths as in reference MobileNet code, prefers rouding up
    rather than down. """
    depths = [32, 16, 24, 40, 80, 96, 192, 320]
    return [_round_to_multiple_of(depth * alpha, 8) for depth in depths]


class MnasMulti(nn.Module):

    def __init__(self, alpha=1.0):
        super(MnasMulti, self).__init__()
        depths = _get_depths(alpha) # channels mobilenet
        if alpha == 1.0:
            # MNASNet = torchvision.models.mnasnet1_0(pretrained=True, progress=True)
            MNASNet = torchvision.models.mnasnet1_0(weights='MNASNet1_0_Weights.IMAGENET1K_V1', progress=True)
        else:
            MNASNet = torchvision.models.MNASNet(alpha=alpha)

        self.conv0 = nn.Sequential(
            MNASNet.layers._modules['0'],
            MNASNet.layers._modules['1'],
            MNASNet.layers._modules['2'],
            MNASNet.layers._modules['3'],
            MNASNet.layers._modules['4'],
            MNASNet.layers._modules['5'],
            MNASNet.layers._modules['6'],
            MNASNet.layers._modules['7'],
            MNASNet.layers._modules['8'],
        )

        self.conv1 = MNASNet.layers._modules['9']
        self.conv2 = MNASNet.layers._modules['10']

        self.out1 = nn.Conv2d(depths[4], depths[4], 1, bias=False) # 80 80
        self.out_channels = [depths[4]] # 80

        final_chs = depths[4] # 80
        self.inner1 = nn.Conv2d(depths[3], final_chs, 1, bias=True) # 40 80
        self.inner2 = nn.Conv2d(depths[2], final_chs, 1, bias=True) # 24 80

        self.out2 = nn.Conv2d(final_chs, depths[3], 3, padding=1, bias=False) # 80 40
        self.out3 = nn.Conv2d(final_chs, depths[2], 3, padding=1, bias=False) # 80 24
        self.out_channels.append(depths[3])
        self.out_channels.append(depths[2]) # [80, 40, 24]

    def forward(self, x): # bs, 3, 480, 640
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)

        # FPN
        intra_feat = conv2
        outputs = []
        out = self.out1(intra_feat)
        outputs.append(out) # bs, 80, 30, 40 H // 16

        intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner1(conv1)
        out = self.out2(intra_feat)
        outputs.append(out) # bs, 40, 60, 80 H // 8

        intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner2(conv0)
        out = self.out3(intra_feat)
        outputs.append(out) # bs, 24, 120, 160 H // 4

        return outputs[::-1] # scale: big[0] -> small[1]


class Encoderdecoder2D(nn.Module):

    def __init__(self, cfg):
        super(Encoderdecoder2D, self).__init__()
        self.pixel_mean = torch.Tensor(cfg.PIXEL_MEAN).view(-1, 1, 1)
        self.pixel_std = torch.Tensor(cfg.PIXEL_STD).view(-1, 1, 1)
        self.arch = cfg.BACKBONE2D.ARC
        self.enable_sem = cfg.NERF.ENABLE_SEM

        # networks
        if cfg.BACKBONE2D.ARC == 'fpn-mnas-1':
            alpha = float(cfg.BACKBONE2D.ARC.split('-')[-1]) # 1.0
            self.backbone2d = MnasMulti(alpha)
        else:
            raise NotImplementedError
        
        if self.enable_sem:
            m2former_cfg = self.setup_model_cfg(cfg.SEMSEG2D.CFG_PATH)
            self.sem_seg = det2_build_model(m2former_cfg)
            self.freeze = cfg.FREEZE_2D
            self.sem_seg.eval()
        return

    def normalizer(self, x):
        """ Normalizes the RGB images to the input range"""
        return (x - self.pixel_mean.type_as(x)) / self.pixel_std.type_as(x)

    def forward(self, images):
        '''
        imgs: bs, v, c, h, w
        '''
        outs = {}

        # image feature extraction
        # in: images; out: feature maps
        if self.arch == 'fpn-mnas-1':
            imgs = torch.unbind(images, 1) # list: [[bs, 3, h, w], ...]
            features = [self.backbone2d(self.normalizer(img)) for img in imgs] # list: [[scale, scale, scale], ...] scale: [bs, c, h, w]
            outs['features'] = features
            outs['sem_seg'] = None
        else:
            raise NotImplementedError

        if self.enable_sem:
            with torch.no_grad():
                self.sem_seg.eval()
                bs, v = images.shape[:2]
                imgs = images.view(-1, *images.shape[2:]) # [bs*v, c, h, w]
                
                out = self.sem_seg([{'image': i} for i in imgs])                 # rgb
                sem_seg = torch.stack([i['sem_seg'] for i in out if 'sem_seg' in i], dim=0) # [bs, c, h, w]
                sem_seg = sem_seg.view(bs, v, *sem_seg.shape[1:]) # [bs, v, c, h, w]
                outs['sem_seg'] = sem_seg
                
        return outs

    @classmethod
    def setup_model_cfg(cls, path): # args
        cfg = get_cfg()
        # add_deeplab_config(cfg)
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        cfg.merge_from_file(path)

        cfg.freeze()
        return cfg


    def load_sem_seg(self, path=None):
        if path:
            DetectionCheckpointer(self.sem_seg).resume_or_load( # , save_dir=args.ckpt_dir
                path, resume=False
            )
            if is_main_process():
                logger.info('Pretrained 2D semantic segmentation model from {}'.format(path))

