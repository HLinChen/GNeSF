import torch
import torch.nn as nn
import functools
from typing import Iterable

from .render import VolumeRender
from utils import sparse_to_dense_channel
from ops.back_project import back_project_batch
from .fpn_fusion import FPNFusion
from .gru_fusion import GRUFusion


def batchify_query(query_fn, *args: Iterable[torch.Tensor], chunk, dim_batchify):
    # [(B), N_rays, N_pts, ...] -> [(B), N_rays*N_pts, ...]
    _N_rays = args[0].shape[dim_batchify]
    _N = args[0].shape[dim_batchify]
    raw_ret = []
    for i in range(0, _N, chunk):
        if dim_batchify == 0:
            args_i = [arg[i:i+chunk] for arg in args]
        elif dim_batchify == 1:
            args_i = [arg[:, i:i+chunk] for arg in args]
        elif dim_batchify == 2:
            args_i = [arg[:, :, i:i+chunk] for arg in args]
        else:
            raise NotImplementedError
        raw_ret_i = query_fn(*args_i)
        if not isinstance(raw_ret_i, tuple):
            raw_ret_i = [raw_ret_i]
        raw_ret.append(raw_ret_i)
    collate_raw_ret = []
    num_entry = 0
    for entry in zip(*raw_ret):
        if isinstance(entry[0], dict):
            tmp_dict = {}
            for list_item in entry:
                for k, v in list_item.items():
                    if k not in tmp_dict:
                        tmp_dict[k] = []
                    tmp_dict[k].append(v)
            for k in tmp_dict.keys():
                # [(B), N_rays*N_pts, ...] -> [(B), N_rays, N_pts, ...]
                # NOTE: compatible with torch 1.6
                v = torch.cat(tmp_dict[k], dim=dim_batchify)
                tmp_dict[k] = v.reshape([*v.shape[:dim_batchify], _N_rays, *v.shape[dim_batchify+1:]])
            entry = tmp_dict
        else:
            # [(B), N_rays*N_pts, ...] -> [(B), N_rays, N_pts, ...]
            v = torch.cat(entry, dim=dim_batchify)
            entry = v.reshape([*v.shape[:dim_batchify], _N_rays, *v.shape[dim_batchify+1:]])
        collate_raw_ret.append(entry)
        num_entry += 1
    if num_entry == 1:
        return collate_raw_ret[0]
    else:
        return tuple(collate_raw_ret)


class RenderNetwork(nn.Module):
    """ 3D network to refine feature volumes"""
    def __init__(self, cfg, attr_ch_in, sdf_pred, out_ch_2d) -> None:
        super().__init__()
        
        self.cfg = cfg
        self.attr_ch_in = attr_ch_in
        self.enable_rgb = cfg.NERF.ENABLE_RGB
        self.enable_sem = cfg.NERF.ENABLE_SEM
        self.render_stride = cfg.NERF.SAMPLE.RENDER_STRIDE
        self.netchunk = cfg.NERF.SAMPLE.NETCHUNK
        self.attr_ch = 0
        if self.enable_rgb:
            self.attr_ch += 3
        if self.enable_sem:
            self.num_cls = cfg.NUM_CLS
            self.attr_ch += self.num_cls
        
        self.k = cfg.NERF.K
        self.fuse_fpn = FPNFusion(out_ch_2d, mode=cfg.NERF.FUSE_FPN_MODE)
        self.render = VolumeRender(cfg, sdf_pred)
        
        if cfg.ENABLE_RENDER:
            ch_sem = cfg.NUM_CLS if self.cfg.NERF.ENABLE_SEM else 0
            ch_rgb = 3 if self.cfg.NERF.ENABLE_RGB else 0
            ch_attr = [ch_sem+ch_rgb]
            self.gru_fusion_attr = GRUFusion(cfg, ch_attr, enable_count=True)
    
    def forward(self, inputs, feats2d, volume, coords, KRcam, interval, volume_tsdf=None, sem_seg=None):
        imgs = inputs['imgs'] / 255                                               # [bs, views, c, h, w]
        positions = inputs['positions']
        origin = inputs['vol_origin_partial']
        rays = inputs['rays']
        rays[..., :3] = rays[..., :3] - origin[:, None]
        
        sem_seg = sem_seg / sem_seg.sum(dim=2, keepdim=True) # prob 对的
        perturb = True if self.training else False
        feats2d = self.fuse_fpn(feats2d, sem_seg)     # [bs, views, c, h//4, w//4] last
        feats2d = torch.cat([imgs, feats2d], dim=2).permute(1, 0, 2, 3, 4).contiguous() # [views, bs, c, h, w]
        
        bs = imgs.shape[0]
        outputs = {}
        losses = {}
        
        vol_attr_pred = torch.zeros(coords.shape[0], 3+self.num_cls, device=volume.device)
        
        for i in range(bs):
            targets = {}
            if self.enable_rgb: targets['rgb'] = inputs['rgb'][i][None]
            if self.enable_sem and 'sem' in inputs: targets['sem'] = inputs['sem'][i][None]
            ray = rays[i][None]
            
            batch_ind = torch.nonzero(coords[:, 0] == i).squeeze(1) # (number of voxels)
            coords_b = (coords[batch_ind, 1:] / interval).trunc().long()
            volume_b = volume[batch_ind]
            
            volume_sdf_b = volume_tsdf[batch_ind]
            volume_sdf_b = sparse_to_dense_channel(coords_b, volume_sdf_b, self.cfg.N_VOX, volume_sdf_b.shape[-1], 1, volume_sdf_b.device) # (n_vox, n_vox, n_vox, c)
            positions_b = positions[i]
            
            volume_attr_b, valid_b = back_project_batch(coords_b, origin[i], self.cfg.VOXEL_SIZE, feats2d[:, i], KRcam[:, i], volume_sdf_b, positions_b) # (number of voxel, c) features/tsdf
            volume_attr_b = volume_attr_b.view(coords_b.shape[0], -1) # (number of voxel, k x c) features/tsdf
            
            attr_pred = self.predict_attr(volume_b, volume_attr_b, valid_b)
            _, attr_pred, _, _ = self.gru_fusion_attr(coords[batch_ind], attr_pred, inputs, 0, counts_in=valid_b.sum(-1, keepdim=True).to(torch.uint8))
            vol_attr_pred[batch_ind] = attr_pred
            
            volume_b = sparse_to_dense_channel(coords_b, volume_b, self.cfg.N_VOX, volume_b.shape[-1], 0, volume_b.device) # (n_vox, n_vox, n_vox, n_views, c)
            volume_attr_pred_b = sparse_to_dense_channel(coords_b, attr_pred, self.cfg.N_VOX, attr_pred.shape[-1], 0, attr_pred.device)
            volume_b = volume_b[None].permute(0, 4, 1, 2, 3).contiguous() # (1, c, n_vox, n_vox, n_vox)
            
            volume_sdf_b = volume_sdf_b[None].permute(0, 4, 1, 2, 3).contiguous() # (1, k x c, n_vox, n_vox, n_vox)
            volume_attr_pred_b = volume_attr_pred_b[None].permute(0, 4, 1, 2, 3).contiguous() # (1, c, n_vox, n_vox, n_vox)
        
            volume_attr_b, valid_b = None, None
            output, loss = self.render(ray, volume_b, volume_attr_b, valid_b, volume_sdf_b, volume_attr_pred=volume_attr_pred_b, perturb=perturb, targets=targets)
            
            
            for k, v in output.items():
                if k not in outputs: outputs[k] = []
                outputs[k].append(v)
                
            for k, v in loss.items():
                if k not in losses: losses[k] = 0
                losses[k] += v
            
        for k, v in outputs.items():
            outputs[k] = torch.cat(v, 0)
        
        if not self.training:
            outputs['vol_attr'] = vol_attr_pred
            
        for k, v in losses.items():
            losses[k] /= bs
            
        return outputs, losses

    def predict_attr(self, feats, attrs, valid): # , attrs, valid
        '''
        feats: n_vox, c
        attrs: n_vox, k x c
        valid: n_vox, k
        '''
        k = valid.shape[-1]
        feats = feats[None]
        attrs = attrs.view(attrs.shape[0], k, -1)[None]
        valid = valid[None]
        
        batchify_query_ = functools.partial(batchify_query, chunk=self.netchunk, dim_batchify=1)
        attrs = batchify_query_(self.render.mlp, feats, attrs[..., :-1], attrs[..., -1], valid)
        
        color = attrs[..., :3]
        if self.enable_sem:
            sem = attrs[..., 3:]
            ans = torch.cat([color, sem], dim=-1)
        else:
            ans = color
        
        return ans.squeeze(0)