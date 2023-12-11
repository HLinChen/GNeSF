# the codes are partly borrowed from IBRNet

import torch
import torch.nn as nn
import torch.nn.functional as F

torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)


# default tensorflow initialization of linear layers
def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)


@torch.jit.script
def fused_mean_variance(x, weight):
    mean = torch.sum(x * weight, dim=-2, keepdim=True)
    var = torch.sum(weight * (x - mean) ** 2, dim=-2, keepdim=True)
    return mean, var


class GeneralRenderingNetwork(nn.Module):
    """
    This model is not sensitive to finetuning
    """

    def __init__(self, in_geometry_feat_ch=8, in_rendering_feat_ch=56, anti_alias_pooling=True, num_clas=0, enable_rgb=True, enable_sem=False):
        super(GeneralRenderingNetwork, self).__init__()

        self.in_geometry_feat_ch = in_geometry_feat_ch
        self.in_rendering_feat_ch = in_rendering_feat_ch
        self.anti_alias_pooling = anti_alias_pooling
        self.rgb_ch = 3 if enable_rgb else 0
        self.sem_ch = num_clas if enable_sem else 0
        self.enable_sem = enable_sem

        if self.anti_alias_pooling:
            self.s = nn.Parameter(torch.tensor(0.2), requires_grad=True)
        activation_func = nn.ELU(inplace=True)

        self.dist_fc_rgb = nn.Sequential(nn.Linear(1, 16),
                                        activation_func,
                                        nn.Linear(16, in_rendering_feat_ch + self.rgb_ch),
                                        activation_func)
        
        # self.base_fc = nn.Sequential(nn.Linear((in_rendering_feat_ch + self.rgb_ch) * 3, 64), #  wo cost volume
        self.base_fc = nn.Sequential(nn.Linear((in_rendering_feat_ch + self.rgb_ch) * 3 + in_geometry_feat_ch, 64), #  + self.sem_ch
                                     activation_func,
                                     nn.Linear(64, 32),
                                     activation_func)

        self.vis_fc = nn.Sequential(nn.Linear(32, 32),
                                    activation_func,
                                    nn.Linear(32, 33),
                                    activation_func,
                                    )

        self.vis_fc2 = nn.Sequential(nn.Linear(32, 32),
                                     activation_func,
                                     nn.Linear(32, 1),
                                     nn.Sigmoid()
                                     )
        
        self.rgb_fc = nn.Sequential(nn.Linear(32 + 1 + 1, 16),
                                    activation_func,
                                    nn.Linear(16, 8),
                                    activation_func,
                                    nn.Linear(8, 1))
        
        if self.enable_sem:
            self.sem_fc = nn.Sequential(nn.Linear(32 + 1 + 1, 16),
                                        activation_func,
                                        nn.Linear(16, 8),
                                        activation_func,
                                        nn.Linear(8, 1))
            self.sem_fc.apply(weights_init)

        self.base_fc.apply(weights_init)
        self.vis_fc2.apply(weights_init)
        self.vis_fc.apply(weights_init)
        # self.attr_fc.apply(weights_init)
        self.rgb_fc.apply(weights_init)

    def forward(self, geometry_feat, attr_feat, dists, mask):
        '''
        :param geometry_feat: geometry features indicates sdf  (1, number of voxel, C)
        :param attr_feat: rgbs sem and image features (1, number of voxel, v, 3+c)
        :param dists: distences between 3d points and camera origin (1, number of voxel, v)
        :param mask: mask for whether each projection is valid or not. (1, number of voxel, v)
        :return: attribute and density output, [1, number of voxel, 4]
        '''
        
        dists = dists[..., None]                                                # (1, number of voxel, v, 1)
        # dists = F.normalize(dists, dim=-2)
        mask = mask[..., None]                                                  # (1, number of voxel, v, 1)
        num_views = attr_feat.shape[-2]                                           
        geometry_feat = geometry_feat[..., None, :].repeat(1, 1, num_views, 1)       # (1, number of voxel, v, C)
        
        rgb_in = attr_feat[..., :self.rgb_ch]
        dist_feat_rgb = self.dist_fc_rgb(dists)
        rgb_chs = [i for i in range(attr_feat.shape[-1]) if i < self.rgb_ch or i >= self.rgb_ch + self.sem_ch]
        rgb_feat = attr_feat[..., rgb_chs] + dist_feat_rgb
        
        if self.enable_sem:
            sem_in = attr_feat[..., self.rgb_ch: self.rgb_ch + self.sem_ch]

        if self.anti_alias_pooling:
            exp_dot_prod = torch.exp(torch.abs(self.s) * (-dists.masked_fill(mask == 0, 1e9)))
            weight = (exp_dot_prod - torch.min(exp_dot_prod, dim=-2, keepdim=True)[0]) * mask
            weight = weight / (torch.sum(weight, dim=-2, keepdim=True) + 1e-8)   # (number of voxel, v, 1)
        else:
            weight = mask / (torch.sum(mask, dim=-2, keepdim=True) + 1e-8)

        # compute mean and variance across different views for each point
        mean, var = fused_mean_variance(rgb_feat, weight)  # [number of voxel, 1, n_feat]
        globalfeat = torch.cat([mean, var], dim=-1)  # [number of voxel, 1, 2*n_feat]

        x = torch.cat([geometry_feat, globalfeat.expand(-1, -1, num_views, -1), rgb_feat], dim=-1)
        x = self.base_fc(x)

        x_vis = self.vis_fc(x * weight)     # [number of voxel, 1, n_feat]
        x_res, vis = torch.split(x_vis, [x_vis.shape[-1] - 1, 1], dim=-1)
        vis = torch.sigmoid(vis) * mask
        x = x + x_res
        vis = self.vis_fc2(x * vis) * mask

        # rgb computation
        x = torch.cat([x, vis, dists], dim=-1)
        x_rgb = self.rgb_fc(x)
        x_rgb = x_rgb.masked_fill(mask == 0, -6e4)
        blending_weights_valid = F.softmax(x_rgb, dim=-2)  # color blending
        rgb_out = torch.sum(rgb_in * blending_weights_valid, dim=-2)
        
        if self.enable_sem:
            x_sem = self.sem_fc(x)
            x_sem = x_sem.masked_fill(mask == 0, -6e4)
            blending_weights_valid = F.softmax(x_sem, dim=-2)  # color blending
            sem_out = torch.sum(sem_in * blending_weights_valid, dim=-2)
        
        attr_out = torch.cat([rgb_out, sem_out], dim=-1) if self.enable_sem else rgb_out
        return attr_out  # (number of voxel, 3)


def build_nerf_mlp(cfg):
    return GeneralRenderingNetwork(
        cfg.NERF.IN_3D_CHANNELS, cfg.NERF.IN_2D_CHANNELS,
        cfg.NERF.ANTI_ALIAS_POOLING, cfg.NUM_CLS,
        cfg.NERF.ENABLE_RGB, cfg.NERF.ENABLE_SEM
    )


