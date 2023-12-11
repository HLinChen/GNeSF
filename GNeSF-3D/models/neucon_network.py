import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsparse.tensor import PointTensor
from loguru import logger

from models.modules import SPVCNN
from utils import apply_log_transform
from .gru_fusion import GRUFusion
from ops.back_project import back_project
from ops.generate_grids import generate_grid
from .render_network import RenderNetwork


class NeuConNet(nn.Module):
    '''
    Coarse-to-fine network.
    '''

    def __init__(self, cfg):
        super(NeuConNet, self).__init__()
        self.cfg = cfg
        self.n_scales = len(cfg.THRESHOLDS) - 1 # 2

        out_ch_2d = [24, 40, 80]
        alpha = int(self.cfg.BACKBONE2D.ARC.split('-')[-1])
        ch_in = [out_ch_2d[2] * alpha + 1, 96 + out_ch_2d[1] * alpha + 2 + 1, 48 + out_ch_2d[0] * alpha + 2 + 1, 24 + 24 + 2 + 1] # [81, 139, 75, 51]
        channels = [96, 48, 24]

        if self.cfg.FUSION.FUSION_ON:
            # GRU Fusion
            self.gru_fusion = GRUFusion(cfg, channels)
            
        # sparse conv
        self.sp_convs = nn.ModuleList() # 3 scales; each scale predict tsdf and occ
        # MLPs that predict tsdf and occupancy.
        self.tsdf_preds = nn.ModuleList()
        self.occ_preds = nn.ModuleList()
        for i in range(len(cfg.THRESHOLDS)):
            self.sp_convs.append(
                SPVCNN(num_classes=1, in_channels=ch_in[i],
                       pres=1,
                       cr=1 / 2 ** i, # 1.0; 0.5; 0.25 (coarse to fine)
                       vres=self.cfg.VOXEL_SIZE * 2 ** (self.n_scales - i), # 0.16; 0.08; 0.04 (coarse to fine)
                       dropout=self.cfg.SPARSEREG.DROPOUT)
            )
            self.tsdf_preds.append(nn.Linear(channels[i], 1))
            self.occ_preds.append(nn.Linear(channels[i], 1))
            
        # nerf
        
        self.render_network = RenderNetwork(cfg, channels[-1], self.tsdf_preds[-1], out_ch_2d) if cfg.ENABLE_RENDER else None

    def get_target(self, coords, inputs, scale):
        '''
        Won't be used when 'fusion_on' flag is turned on
        :param coords: (Tensor), coordinates of voxels, (N, 4) (4 : Batch ind, x, y, z)
        :param inputs: (List), inputs['tsdf_list' / 'occ_list']: ground truth volume list, [(B, DIM_X, DIM_Y, DIM_Z)]
        :param scale:
        :return: tsdf_target: (Tensor), tsdf ground truth for each predicted voxels, (N,)
        :return: occ_target: (Tensor), occupancy ground truth for each predicted voxels, (N,)
        '''
        with torch.no_grad():
            tsdf_target = inputs['tsdf_list'][scale]
            occ_target = inputs['occ_list'][scale]
            coords_down = coords.detach().clone().long()
            coords_down[:, 1:] = torch.div(coords[:, 1:], 2 ** scale, rounding_mode='trunc')
            tsdf_target = tsdf_target[coords_down[:, 0], coords_down[:, 1], coords_down[:, 2], coords_down[:, 3]]
            occ_target = occ_target[coords_down[:, 0], coords_down[:, 1], coords_down[:, 2], coords_down[:, 3]]
            return tsdf_target, occ_target

    def upsample(self, pre_feat, pre_coords, interval, num=8):
        '''

        :param pre_feat: (Tensor), features from last level, (N, C)
        :param pre_coords: (Tensor), coordinates from last level, (N, 4) (4 : Batch ind, x, y, z)
        :param interval: interval of voxels, interval = scale ** 2
        :param num: 1 -> 8
        :return: up_feat : (Tensor), upsampled features, (N*8, C)
        :return: up_coords: (N*8, 4), upsampled coordinates, (4 : Batch ind, x, y, z)
        '''
        with torch.no_grad():
            pos_list = [1, 2, 3, [1, 2], [1, 3], [2, 3], [1, 2, 3]]
            n, c = pre_feat.shape
            up_feat = pre_feat.unsqueeze(1).expand(-1, num, -1).contiguous()
            up_coords = pre_coords.unsqueeze(1).repeat(1, num, 1).contiguous()
            for i in range(num - 1):
                up_coords[:, i + 1, pos_list[i]] += interval

            up_feat = up_feat.view(-1, c)
            up_coords = up_coords.view(-1, 4)

        return up_feat, up_coords

    def forward(self, features, inputs, outputs, sem_seg=None):
        '''
        features: [[view, bs, c, h, w], ...] multi-scale
        :param features: list: features for each image: eg. list[0] : pyramid features for image0 : [(B, C0, H, W), (B, C1, H/2, W/2), (B, C2, H/2, W/2)]
        :param inputs: meta data from dataloader
        :param outputs: {}
        :return: outputs: dict: {
            'coords':                  (Tensor), coordinates of voxels,
                                    (number of voxels, 4) (4 : batch ind, x, y, z)
            'tsdf':                    (Tensor), TSDF of voxels,
                                    (number of voxels, 1)
        }
        :return: loss_dict: dict: {
            'tsdf_occ_loss_X':         (Tensor), multi level loss
        }
        '''
        bs = features[0][0].shape[0]
        pre_feat = None
        pre_coords = None
        loss_dict = {}
        # ----coarse to fine----
        for i in range(self.cfg.N_LAYER):
            interval = 2 ** (self.n_scales - i) # 4 -> 2 -> 1
            scale = self.n_scales - i

            if i == 0:
                # ----generate new coords---- local
                coords = generate_grid(self.cfg.N_VOX, interval)[0] # [3, n_vox//interval x n_vox//interval x n_vox//interval]
                up_coords = []
                for b in range(bs):
                    up_coords.append(torch.cat([torch.ones(1, coords.shape[-1]).to(coords.device) * b, coords]))
                up_coords = torch.cat(up_coords, dim=1).permute(1, 0).contiguous() # (num of voxels x bs, 4) (4 : batch ind, x, y, z)
            else:
                # ----upsample coords----
                up_feat, up_coords = self.upsample(pre_feat, pre_coords, interval)

            # ----back project----
            feats = torch.stack([feat[scale] for feat in features]) # n_views, bs, c, h, w
            KRcam = inputs['proj_matrices'][:, :, scale].permute(1, 0, 2, 3).contiguous() # n_views, bs, 4, 4
            volume, count = back_project(up_coords, inputs['vol_origin_partial'], self.cfg.VOXEL_SIZE, feats, # 2D feature -> 3D feature local # [number of voxel x bs, c+1]
                                         KRcam)
            grid_mask = count > 1 # [number of voxel]

            # ----concat feature from last stage----
            if i != 0:
                feat = torch.cat([volume, up_feat], dim=1)
            else:
                feat = volume # [num of voxels x bs, c+1]

            if not self.cfg.FUSION.FUSION_ON: # when not fusing, get ground truth
                tsdf_target, occ_target = self.get_target(up_coords, inputs, scale)

            # ----convert to aligned camera coordinate----
            r_coords = up_coords.detach().clone().float() # (num of voxels x bs, 4)
            for b in range(bs):
                batch_ind = torch.nonzero(up_coords[:, 0] == b).squeeze(1) # (num of voxels)
                coords_batch = up_coords[batch_ind][:, 1:].float() # get a batch of up_coords 
                coords_batch = coords_batch * self.cfg.VOXEL_SIZE + inputs['vol_origin_partial'][b].float() # grid -> coord
                coords_batch = torch.cat((coords_batch, torch.ones_like(coords_batch[:, :1])), dim=1) # (num of voxels, 4)
                coords_batch = coords_batch @ inputs['world_to_aligned_camera'][b, :3, :].permute(1, 0).contiguous() # ref sys (num of voxels, 4)
                r_coords[batch_ind, 1:] = coords_batch # (num of voxels x bs, 4)

            # batch index is in the last position
            r_coords = r_coords[:, [1, 2, 3, 0]]

            # ----sparse conv 3d backbone----
            point_feat = PointTensor(feat, r_coords) # point + feature {}
            feat = self.sp_convs[i](point_feat) # (num of voxels x bs, c')

            # ----gru fusion----
            if self.cfg.FUSION.FUSION_ON:
                up_coords, feat, tsdf_target, occ_target = self.gru_fusion(up_coords, feat, inputs, i)
                if self.cfg.FUSION.FULL:
                    grid_mask = torch.ones_like(feat[:, 0]).bool()

            tsdf = self.tsdf_preds[i](feat) # (num of voxels x bs, 1)
            occ = self.occ_preds[i](feat) # (num of voxels x bs, 1)

            # -------compute loss-------
            if self.render_network is None:
                if tsdf_target is not None:
                    loss = self.compute_loss(tsdf, occ, tsdf_target, occ_target,
                                            mask=grid_mask,
                                            pos_weight=self.cfg.POS_WEIGHT)
                loss_dict.update({f'tsdf_occ_loss_{i}': loss})

            # ------define the sparsity for the next stage-----
            occupancy = occ.squeeze(1) > self.cfg.THRESHOLDS[i] # (num of voxels x bs)
            occupancy[grid_mask == False] = False

            num = int(occupancy.sum().data.cpu())

            if num == 0:
                logger.warning('no valid points: scale {}'.format(i))
                return outputs, loss_dict

            # ------avoid out of memory: sample points if num of points is too large-----
            if self.training and num > self.cfg.TRAIN_NUM_SAMPLE[i] * bs:
                choice = np.random.choice(num, num - self.cfg.TRAIN_NUM_SAMPLE[i] * bs,
                                          replace=False)
                ind = torch.nonzero(occupancy)
                occupancy[ind[choice]] = False

            pre_coords = up_coords[occupancy]
            for b in range(bs):
                batch_ind = torch.nonzero(pre_coords[:, 0] == b).squeeze(1)
                if len(batch_ind) == 0:
                    logger.warning('no valid points: scale {}, batch {}'.format(i, b))
                    return outputs, loss_dict

            pre_feat = feat[occupancy]
            pre_tsdf = tsdf[occupancy]
            pre_occ = occ[occupancy]

            pre_feat = torch.cat([pre_feat, pre_tsdf, pre_occ], dim=1) # [num of voxels, c]

            if i == self.cfg.N_LAYER - 1:
                if self.cfg.ENABLE_MESH_GT:
                    occ_target = occ_target.squeeze(1)
                    pre_feat = feat[occ_target]
                    pre_coords = up_coords[occ_target]
                    pre_tsdf = tsdf_target[occ_target]
                    pre_occ = occ_target[occ_target][..., None]
                    pre_feat = torch.cat([pre_feat, pre_tsdf, pre_occ], dim=1) # [num of voxels, c]
                
                outputs['coords'] = pre_coords
                outputs['tsdf'] = pre_tsdf
                

        # if self.enable_render: 
        if self.render_network is not None: 
            KRcam[:, :, :2] *= 4
            inputs.pop('occ_list')
            output, loss = self.render_network(inputs, features, pre_feat[:, :-2], pre_coords, KRcam, interval, pre_tsdf, sem_seg)
            if not self.training:
                vol_attr = output.pop('vol_attr')
                rgb =vol_attr[:, :3]
                sem = vol_attr[:, 3:].argmax(dim=-1)[..., None]
                outputs['tsdf'] = torch.cat([outputs['tsdf'], rgb, sem], dim=-1)
            outputs.update(output)
            loss_dict.update(loss)

        return outputs, loss_dict

    @staticmethod
    def compute_loss(tsdf, occ, tsdf_target, occ_target, loss_weight=(1, 1),
                     mask=None, pos_weight=1.0):
        '''

        :param tsdf: (Tensor), predicted tsdf, (num of voxels x bs, 1)
        :param occ: (Tensor), predicted occupancy, (num of voxels x bs, 1)
        :param tsdf_target: (Tensor),ground truth tsdf, (num of voxels x bs, 1)
        :param occ_target: (Tensor), ground truth occupancy, (num of voxels x bs, 1)
        :param loss_weight: (Tuple)
        :param mask: (Tensor), mask voxels which cannot be seen by all views (num of voxels)
        :param pos_weight: (float)
        :return: loss: (Tensor)
        '''
        # compute occupancy/tsdf loss
        tsdf = tsdf.view(-1) # (num of voxels)
        occ = occ.view(-1) # (num of voxels)
        tsdf_target = tsdf_target.view(-1) # (num of voxels)
        occ_target = occ_target.view(-1) # (num of voxels)
        if mask is not None:
            mask = mask.view(-1) # (num of voxels)
            tsdf = tsdf[mask]
            occ = occ[mask]
            tsdf_target = tsdf_target[mask]
            occ_target = occ_target[mask]

        n_all = occ_target.shape[0]
        n_p = occ_target.sum()
        if n_p == 0:
            logger.warning('target: no valid voxel when computing loss')
            return torch.Tensor([0.0]).cuda()[0] * tsdf.sum()
        w_for_1 = (n_all - n_p).float() / n_p
        w_for_1 *= pos_weight

        # compute occ bce loss
        occ_loss = F.binary_cross_entropy_with_logits(occ, occ_target.float(), pos_weight=w_for_1)

        # compute tsdf l1 loss
        tsdf = apply_log_transform(tsdf[occ_target])
        tsdf_target = apply_log_transform(tsdf_target[occ_target])
        tsdf_loss = torch.mean(torch.abs(tsdf - tsdf_target))

        # compute final loss
        loss = loss_weight[0] * occ_loss + loss_weight[1] * tsdf_loss
        return loss
