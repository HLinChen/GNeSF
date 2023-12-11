import functools
from typing import Iterable
from collections import OrderedDict

import torch
import numpy as np
from torch import nn
from torch import autograd
import torch.nn.functional as F

from .mlp_network import build_nerf_mlp
from .ray_sampler import fine_sample, sdf2weights



def batchify_query(query_fn, *args: Iterable[torch.Tensor], chunk, dim_batchify, pre_sdf=True):
    # [(B), N_rays, N_pts, ...] -> [(B), N_rays*N_pts, ...]
    _N_rays = args[0].shape[dim_batchify]
    _N_pts = args[0].shape[dim_batchify+1]
    args = [arg.flatten(dim_batchify, dim_batchify+1) for arg in args]
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
        raw_ret_i = query_fn(*args_i, pre_sdf)
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
                # tmp_dict[k] = torch.cat(tmp_dict[k], dim=dim_batchify).unflatten(dim_batchify, [_N_rays, _N_pts])
                # NOTE: compatible with torch 1.6
                v = torch.cat(tmp_dict[k], dim=dim_batchify)
                tmp_dict[k] = v.reshape([*v.shape[:dim_batchify], _N_rays, _N_pts, *v.shape[dim_batchify+1:]])
            entry = tmp_dict
        else:
            # [(B), N_rays*N_pts, ...] -> [(B), N_rays, N_pts, ...]
            # entry = torch.cat(entry, dim=dim_batchify).unflatten(dim_batchify, [_N_rays, _N_pts])
            # NOTE: compatible with torch 1.6
            v = torch.cat(entry, dim=dim_batchify)
            entry = v.reshape([*v.shape[:dim_batchify], _N_rays, _N_pts, *v.shape[dim_batchify+1:]])
        collate_raw_ret.append(entry)
        num_entry += 1
    if num_entry == 1:
        return collate_raw_ret[0]
    else:
        return tuple(collate_raw_ret)


class VolumeRender(nn.Module):
    """ 3D network to refine feature volumes"""

    def __init__(self, cfg, sdf_net): # , mlp
        super(VolumeRender, self).__init__()
        
        self.rayschunk = cfg.NERF.SAMPLE.RAYSCHUNK
        self.netchunk = cfg.NERF.SAMPLE.NETCHUNK
        self.netchunk_val = cfg.NERF.SAMPLE.NETCHUNK_VAL
        self.N_samples = cfg.NERF.SAMPLE.N_SAMPLES
        self.N_importance = cfg.NERF.SAMPLE.N_IMPORTANCE
        self.max_upsample_steps = cfg.NERF.SAMPLE.MAX_UPSAMPLE_STEPS
        self.max_bisection_steps = cfg.NERF.SAMPLE.MAX_BISECTION_STEPS
        self.epsilon = cfg.NERF.SAMPLE.EPSILON
        self.near = cfg.NERF.NEAR
        self.far = cfg.NERF.FAR

        self.speed_factor = cfg.NERF.SPEED_FACTOR
        ln_beta_init = np.log(cfg.NERF.BETA_INIT) / self.speed_factor
        self.ln_beta = nn.Parameter(data=torch.Tensor([ln_beta_init]), requires_grad=True)
        
        # self.mlp = mlp
        self.mlp = build_nerf_mlp(cfg)
        self.sdf_net = sdf_net
        self.enable_rgb = cfg.NERF.ENABLE_RGB
        self.enable_sem = cfg.NERF.ENABLE_SEM
        self.voxel_size = cfg.VOXEL_SIZE
        self.voxel_dim = cfg.N_VOX
        self.label_smoothing = 1 # cfg.HEADS3D.TSDF.LABEL_SMOOTHING
        self.ignore_label = cfg.IGNORE_LABEL
        
        self.num_cls = cfg.NUM_CLS
        
    def forward(self, rays, volume, volume_attr, volume_attr_valid, volume_sdf, volume_attr_pred=None, perturb=True, targets=None):
        # bs, c, nx, ny, nz = volume.shape
        self.volume = volume
        # self.volume_attr = volume_attr
        # self.volume_attr_valid = volume_attr_valid.float() # [bs, k, nx, ny, nz] # .view(bs, nx, ny, nz, -1).permute(0, 4, 1, 2, 3)
        
        self.volume_attr_pred = volume_attr_pred
        self.volume_sdf = volume_sdf
        
        rays_o, rays_d = rays[:, :, :3], rays[:, :, 3:6]
        rays_d[rays_d.abs() < 1e-6] = 1e-6

        self.device = rays_o.device
        self.perturb = perturb
        
        DIM_BATCHIFY = 1
        self.B = rays_d.shape[0]  # batch_size
        flat_vec_shape = [self.B, -1, 3]

        rays_o = torch.reshape(rays_o, flat_vec_shape).float()
        rays_d = torch.reshape(rays_d, flat_vec_shape).float()

        depth_ratio = rays_d.norm(dim=-1)
        rays_d = F.normalize(rays_d, dim=-1)
        
        netchunk = self.netchunk if self.training else self.netchunk_val #  * 128 # 512
        
        self.batchify_query_ = functools.partial(batchify_query, chunk=netchunk, dim_batchify=DIM_BATCHIFY)

        output = {}
        losses = {}
        for i in range(0, rays_o.shape[DIM_BATCHIFY], self.rayschunk):
            ret_i = self.render_rayschunk(rays_o[:, i:i+self.rayschunk], rays_d[:, i:i+self.rayschunk], depth_ratio[:, i:i+self.rayschunk]) # volume, volume_attr, volume_attr_valid, 
            for k, v in ret_i.items():
                if k not in output:
                    output[k] = []
                output[k].append(v)
        for k, v in output.items():
            output[k] = torch.cat(v, DIM_BATCHIFY)
        
        del self.batchify_query_ # depth_ratio, 
        
        # compute losses
        if targets is not None:
            losses = self.compute_loss(output, targets)

        return output, losses

    def render_rayschunk(self, rays_o, rays_d, depth_ratio):
        prefix_batch = [self.B]
        N_rays = rays_o.shape[-2]
        
        nears = self.near * torch.ones([*prefix_batch, N_rays, 1]).to(self.device)    # b, n_rays, 1
        fars = self.far * torch.ones([*prefix_batch, N_rays, 1]).to(self.device)

        _t = torch.linspace(0, 1, self.N_samples).float().to(self.device)
        d_coarse = nears * (1 - _t) + fars * _t                             # b, n_rays, N_samples
        if self.N_importance > 0:
            alpha, beta = self.forward_ab()
            with torch.no_grad():
                _t = torch.linspace(0, 1, self.N_samples*4).float().to(self.device)
                d_init = nears * (1 - _t) + fars * _t
                
                d_fine, beta_map, iter_usage = fine_sample(
                    self.pre_sdf, d_init, rays_o, rays_d, 
                    alpha_net=alpha, beta_net=beta, far=fars, 
                    eps=self.epsilon, max_iter=self.max_upsample_steps, max_bisection=self.max_bisection_steps, 
                    final_N_importance=self.N_importance, perturb=self.perturb, 
                    N_up=self.N_samples*4
                )

            d_all = torch.cat([d_coarse, d_fine], dim=-1)                               # b, n_rays, 1
        else:
            d_all = d_coarse
        d_all, _ = torch.sort(d_all, dim=-1)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * d_all[..., :, None]     # b, n_rays, N_samples, 3
        
        ans = self.batchify_query_(self.pre_attr, pts, pre_sdf=True) # , volume, volume_attr, volume_attr_valid
        attrs, sdf = ans[:2]
        
        valid = (sdf.sum(-1) != sdf.shape[-1])
        
        if self.enable_rgb: radiances = attrs[..., :3]
        if self.enable_sem: sem = attrs[..., 3:]
        
        tau_i = sdf2weights(sdf, d_all)
        rgb_map = torch.sum(tau_i[..., None] * radiances, dim=-2)  # b, n_rays, 3
        distance_map = torch.sum(tau_i / (tau_i.sum(-1, keepdim=True)+1e-10) * d_all, dim=-1)
        
        if self.enable_sem: sem_map = torch.sum(tau_i[..., None] * sem, dim=-2)  # b, n_rays, 3
        
        depth_map = distance_map / depth_ratio # self.
        # acc_map = torch.sum(tau_i, -1)

        ret_i = OrderedDict([
            ('rgb', rgb_map),
            # ('semantic', semantic_map),
            # ('distance', distance_map),
            ('depth', depth_map),
            ('valid', valid),
            # ('mask_volume', acc_map)
        ])
        
        if self.enable_sem: ret_i['sem'] = sem_map


        return ret_i

    def forward_ab(self):
        beta = torch.exp(self.ln_beta * self.speed_factor)
        return 1./beta, beta

    def compute_loss(self, output, targets):
        res = {}
        valid = output.get('valid', None)
        if self.enable_rgb:
            trg_rgb = targets['rgb'] if valid is not None else targets['rgb'][valid][None]
            pre_rgb = output['rgb'] if valid is not None else output['rgb'][valid][None]
            rgb_loss = F.l1_loss(pre_rgb, trg_rgb, reduction='none').mean() # * 10 # Eq.5
            res['rgb_loss'] = rgb_loss
        
        if self.enable_sem and 'sem' in targets:
            trg_sem = targets['sem'].view(-1) if valid is not None else targets['sem'][valid].view(-1)
            pre_sem = output['sem'].view(-1, self.num_cls) if valid is not None else output['sem'][valid].view(-1, self.num_cls)
            # sem_loss = F.cross_entropy(pre_sem, trg_sem, ignore_index=self.ignore_label) # , reduction='none' .mean()
            # sem_loss = F.nll_loss(torch.log(pre_sem), trg_sem, ignore_index=self.ignore_label)
            sem_loss = 1 + F.nll_loss(pre_sem, trg_sem, ignore_index=self.ignore_label)
            res['sem_loss'] = sem_loss
        
        return res

    def pre_attr(self, x, pre_sdf=False): # , volume, volume_attr, volume_attr_valid
        '''
        x: b, n_rays, N_samples, 3
        volume: b, c, x, y, z
        '''
        
        x, valid_coord = self.normolize_pts(x)
        x = x[..., None, None, :]   # b, n_pts, 1, 1, 3
        
        attrs = F.grid_sample(self.volume_attr_pred, x, mode='bilinear', padding_mode='zeros', align_corners = False) # ????  b, c, n_pts, 1, 1
        attrs = attrs.squeeze(-1).squeeze(-1).permute(0, 2, 1) # b, n_pts, c
        
        if pre_sdf:
            sdf = self.pre_sdf(x, do_grid_sample=False, valid=valid_coord)
            
        return attrs, sdf # attrs, sdf #, nabla
    
    def normolize_pts(self, x):
        voxel_dim = torch.tensor(self.voxel_dim).to(x.device)
        
        nxyz = voxel_dim * self.voxel_size
        
        x = 2 * (x / nxyz) - 1 # normalize to [-1, 1] b, n_rays, N_samples, 3
        x = x.flip(-1)
        
        valid = (x.abs() <= 1).all(-1)
        
        return x, valid
    
    def pre_sdf(self, x, do_grid_sample=True, valid=None):
        '''
        x: b, n_rays, N_samples, 3
        volume: b, c, x, y, z
        '''
        
        if do_grid_sample: 
            x, valid = self.normolize_pts(x)
            x = x[..., None, :]         # b, n_rays, N_samples, 1, 3
            
        sdf = F.grid_sample(self.volume_sdf, x, mode='bilinear', padding_mode='border', align_corners = True) # ????  b, c, n_pts, 1, 1
        sdf = sdf.squeeze(-1).squeeze(-1).squeeze(1)
        
        return sdf
    
    def forward_with_nablas(self, x: torch.Tensor, has_grad_bypass: bool = None):
        has_grad = torch.is_grad_enabled() if has_grad_bypass is None else has_grad_bypass
        # force enabling grad for normal calculation
        with torch.enable_grad():
            x = x.requires_grad_(True)
            sdf = self.pre_sdf(x, do_grid_sample=False)
            nabla = autograd.grad(
                sdf,
                x,
                torch.ones_like(sdf, device=x.device),
                create_graph=has_grad,
                retain_graph=has_grad,
                only_inputs=True
            )[0]
        if not has_grad:
            sdf = sdf.detach()
            nabla = nabla.detach()
        nabla = nabla.squeeze(-2).squeeze(-2)
        return sdf, nabla # , h
    