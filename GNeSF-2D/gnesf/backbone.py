
import torch
from loguru import logger
from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model as det2_build_model
import torch.nn as nn
import torch.nn.functional as F

from detectron2.projects.deeplab import add_deeplab_config
from seg2d.mask2former import add_maskformer2_config

from gnesf.feature_network import ResUNet
from tool.comm import is_main_process


class Encoderdecoder2D(nn.Module):

    def __init__(self, args):
        super(Encoderdecoder2D, self).__init__()
        self.args = args
        self.feature_net = ResUNet(coarse_out_ch=self.args.coarse_feat_dim,
                                fine_out_ch=self.args.fine_feat_dim,
                                coarse_only=self.args.coarse_only).cuda()
        
        if args.enable_semantic:
            m2former_cfg = self.setup_model_cfg(args.seg2d_model_cfg)
            self.sem_seg = det2_build_model(m2former_cfg) # .to(self.device)
            self.sem_seg.eval()
            for i in self.sem_seg.parameters(): i.requires_grad = False
            self.load_sem_seg(args.seg2d_model_path)
        
        return

    def forward(self, images):
        '''
        imgs: bs, v, c, h, w
        '''
        outs = []
        
        featmaps = self.feature_net(images)
        outs.append(featmaps)

        if self.args.enable_semantic:
            with torch.no_grad():
                self.sem_seg.eval()
                
                out = self.sem_seg([{'image': i} for i in images*255])                 # rgb
                sem_seg = torch.stack([i['sem_seg'] for i in out if 'sem_seg' in i], dim=0) # [bs*v, c, h, w]
                
                sem_seg = sem_seg / sem_seg.sum(1, keepdim=True)
                
                if self.args.enable_one_hot:
                    sem_seg = sem_seg.argmax(1)
                
                if self.args.enable_one_hot:
                    sem_seg = F.one_hot(sem_seg, num_classes=self.args.num_cls).permute(0, 3, 1, 2).float()
                
                outs.append(sem_seg)
                
        
        return outs

    @classmethod
    def setup_model_cfg(cls, path): # args
        cfg = get_cfg()
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

