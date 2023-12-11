import torch
import torch.nn as nn
from torchsparse.tensor import PointTensor
from utils import sparse_to_dense_channel, sparse_to_dense_torch
from .modules import ConvGRU


class GRUFusion(nn.Module):
    """
    Two functionalities of this class:
    1. GRU Fusion module as in the paper. Update hidden state features with ConvGRU.
    2. Substitute TSDF in the global volume when direct_substitute = True.
    """

    def __init__(self, cfg, ch_in=None, direct_substitute=False, enable_count=False):
        super(GRUFusion, self).__init__()
        self.cfg = cfg
        # replace tsdf in global tsdf volume by direct substitute corresponding voxels
        self.direct_substitude = direct_substitute

        if direct_substitute:
            # tsdf
            self.ch_in = [1, 1, 1] if ch_in is None else ch_in
            self.feat_init = 1
        else:
            # features
            self.ch_in = ch_in
            self.feat_init = 0

        self.enable_count = enable_count
        self.n_layer = len(self.ch_in)
        self.n_scales = len(cfg.THRESHOLDS) - 1
        self.scene_name = [None] * len(self.ch_in)
        self.global_origin = [None] * len(self.ch_in)
        self.global_volume = [None] * len(self.ch_in)
        self.target_tsdf_volume = [None] * len(self.ch_in)
        self.global_counts = [None] * len(self.ch_in) if self.enable_count else None

        if direct_substitute or self.enable_count:
            self.fusion_nets = None
        else:
            self.fusion_nets = nn.ModuleList()
            for i, ch in enumerate(ch_in):
                self.fusion_nets.append(ConvGRU(hidden_dim=ch,
                                                input_dim=ch,
                                                pres=1,
                                                vres=self.cfg.VOXEL_SIZE * 2 ** (self.n_scales - i)))

    def reset(self, i):
        self.global_volume[i] = PointTensor(torch.Tensor([]), torch.Tensor([]).view(0, 3).long()).cuda()
        self.target_tsdf_volume[i] = PointTensor(torch.Tensor([]), torch.Tensor([]).view(0, 3).long()).cuda()
        
        if self.enable_count:
            self.global_counts[i] = torch.Tensor([]).to(torch.uint8).cuda()

    def convert2dense(self, current_coords, current_values, coords_target_global, tsdf_target, relative_origin,
                      scale, current_counts=None):
        '''
        1. convert sparse feature to dense feature;
        2. combine current feature coordinates and previous coordinates within FBV from global hidden state to get
        new feature coordinates (updated_coords);
        3. fuse ground truth tsdf.

        :param current_coords: (Tensor), current coordinates, (N, 3) camera sys
        :param current_values: (Tensor), current features/tsdf, (N, C)
        :param coords_target_global: (Tensor), ground truth coordinates, (N', 3)
        :param tsdf_target: (Tensor), tsdf ground truth, (N',)
        :param relative_origin: (Tensor), origin in global volume, (3,)
        :param scale:
        :return: updated_coords: (Tensor), coordinates after combination, (N', 3)
        :return: current_volume: (Tensor), current dense feature/tsdf volume, (DIM_X, DIM_Y, DIM_Z, C)
        :return: global_volume: (Tensor), global dense feature/tsdf volume, (DIM_X, DIM_Y, DIM_Z, C)
        :return: target_volume: (Tensor), dense target tsdf volume, (DIM_X, DIM_Y, DIM_Z, 1)
        :return: valid: mask: 1 represent in current FBV (N,)
        :return: valid_target: gt mask: 1 represent in current FBV (N,)
        '''
        # previous frame
        global_coords = self.global_volume[scale].C
        global_value = self.global_volume[scale].F
        global_counts = self.global_counts[scale] if self.enable_count else None
        global_tsdf_target = self.target_tsdf_volume[scale].F
        global_coords_target = self.target_tsdf_volume[scale].C

        # dim = (torch.Tensor(self.cfg.N_VOX).cuda() // 2 ** (self.n_layer - scale - 1)).int()
        dim = torch.div(torch.Tensor(self.cfg.N_VOX).cuda(), 2 ** (self.n_layer - scale - 1), rounding_mode='trunc').int()
        dim_list = dim.data.cpu().numpy().tolist()

        # mask voxels that are out of the FBV
        global_coords = global_coords - relative_origin # transfer to local coord; int grid
        valid = ((global_coords < dim) & (global_coords >= 0)).all(dim=-1) # valid global coords
        if self.cfg.FUSION.FULL is False:
            valid_volume = sparse_to_dense_torch(current_coords, 1, dim_list, 0, global_value.device)
            value = valid_volume[global_coords[valid][:, 0], global_coords[valid][:, 1], global_coords[valid][:, 2]]
            all_true = valid[valid]
            all_true[value == 0] = False
            valid[valid] = all_true
        # sparse to dense; local coord
        global_volume = sparse_to_dense_channel(global_coords[valid], global_value[valid], dim_list, self.ch_in[scale], # (H, W, Z, c)
                                                self.feat_init, global_value.device)

        current_volume = sparse_to_dense_channel(current_coords, current_values, dim_list, self.ch_in[scale], # (H, W, Z, c)
                                                 self.feat_init, global_value.device)
        
        global_counts_volume = sparse_to_dense_channel(global_coords[valid], global_counts[valid], dim_list, 1, 0, global_counts.device) if self.enable_count else None
        
        current_counts_volume = sparse_to_dense_channel(current_coords, current_counts, dim_list, 1, 0, current_counts.device) if self.enable_count else None


        if self.cfg.FUSION.FULL is True and self.n_layer != 1:
            # change the structure of sparsity, combine current coordinates and previous coordinates from global volume
            if self.direct_substitude:
                updated_coords = torch.nonzero((global_volume.abs() < 1).any(-1) | (current_volume.abs() < 1).any(-1)) # tsdf
            else:
                updated_coords = torch.nonzero((global_volume != 0).any(-1) | (current_volume != 0).any(-1)) # coords where has feature (number of voxel, 3)
        else:
            updated_coords = current_coords

        # fuse ground truth
        if tsdf_target is not None:
            # mask voxels that are out of the FBV
            global_coords_target = global_coords_target - relative_origin # local coords
            valid_target = ((global_coords_target < dim) & (global_coords_target >= 0)).all(dim=-1)
            # combine current tsdf and global tsdf
            coords_target = torch.cat([global_coords_target[valid_target], coords_target_global])[:, :3] # (number of voxel, 3)
            tsdf_target = torch.cat([global_tsdf_target[valid_target], tsdf_target.unsqueeze(-1)])
            # sparse to dense
            target_volume = sparse_to_dense_channel(coords_target, tsdf_target, dim_list, 1, 1, # (H, W, Z, 1)
                                                    tsdf_target.device)
        else:
            target_volume = valid_target = None

        return updated_coords, current_volume, global_volume, target_volume, valid, valid_target, current_counts_volume, global_counts_volume

    def update_map(self, value, coords, target_volume, valid, valid_target,
                   relative_origin, scale, counts=None):
        '''
        Replace Hidden state/tsdf in global Hidden state/tsdf volume by direct substitute corresponding voxels
        :param value: (Tensor) fused feature (N, C)
        :param coords: (Tensor) updated coords (N, 3)
        :param target_volume: (Tensor) tsdf volume (DIM_X, DIM_Y, DIM_Z, 1)
        :param valid: (Tensor) mask: 1 represent in current FBV (N,)
        :param valid_target: (Tensor) gt mask: 1 represent in current FBV (N,)
        :param relative_origin: (Tensor), origin in global volume, (3,)
        :param scale:
        :return:
        '''
        # pred
        self.global_volume[scale].F = torch.cat(
            [self.global_volume[scale].F[valid == False], value])
        coords = coords + relative_origin
        self.global_volume[scale].C = torch.cat([self.global_volume[scale].C[valid == False], coords])
        
        if self.enable_count:
            self.global_counts[scale] = torch.cat([self.global_counts[scale][valid == False], counts])

        # target
        if target_volume is not None:
            target_volume = target_volume.squeeze()
            self.target_tsdf_volume[scale].F = torch.cat(
                [self.target_tsdf_volume[scale].F[valid_target == False],
                 target_volume[target_volume.abs() < 1].unsqueeze(-1)])
            target_coords = torch.nonzero(target_volume.abs() < 1) + relative_origin

            self.target_tsdf_volume[scale].C = torch.cat(
                [self.target_tsdf_volume[scale].C[valid_target == False], target_coords])

    def save_mesh(self, scale, outputs, scene):
        if outputs is None:
            outputs = dict()
        if "scene_name" not in outputs:
            outputs['origin'] = []
            outputs['scene_tsdf'] = []
            outputs['scene_name'] = []
            if self.cfg.NERF.ENABLE_RGB:
                outputs['scene_rgb'] = []
            if self.cfg.NERF.ENABLE_SEM:
                outputs['scene_sem'] = []
        # only keep the newest result
        if scene in outputs['scene_name']:
            # delete old
            idx = outputs['scene_name'].index(scene)
            del outputs['origin'][idx]
            del outputs['scene_tsdf'][idx]
            del outputs['scene_name'][idx]

        # scene name
        outputs['scene_name'].append(scene)

        fuse_coords = self.global_volume[scale].C
        volume = self.global_volume[scale].F.squeeze(-1)
        max_c = torch.max(fuse_coords, dim=0)[0][:3]
        min_c = torch.min(fuse_coords, dim=0)[0][:3]
        outputs['origin'].append(min_c * self.cfg.VOXEL_SIZE * (2 ** (self.n_layer - scale - 1)))

        ind_coords = fuse_coords - min_c
        dim_list = (max_c - min_c + 1).int().data.cpu().numpy().tolist()
        tsdf = volume[..., 0]
        tsdf_volume = sparse_to_dense_torch(ind_coords, tsdf, dim_list, 1, tsdf.device)
        outputs['scene_tsdf'].append(tsdf_volume)
        if self.cfg.NERF.ENABLE_RGB:
            rgb = (volume[..., 1:4] * 255).round().clip(0, 255).to(torch.uint8)
            rgb_volume = sparse_to_dense_channel(ind_coords, rgb, dim_list, 3, 255, tsdf.device)
            outputs['scene_rgb'].append(rgb_volume)
        if self.cfg.NERF.ENABLE_SEM:
            sem = volume[..., -1].to(torch.uint8)
            sem_volume = sparse_to_dense_torch(ind_coords, sem, dim_list, 255, tsdf.device)
            outputs['scene_sem'].append(sem_volume)

        return outputs

    def forward(self, coords, values_in, inputs, scale=2, outputs=None, save_mesh=False, counts_in=None):
        '''
        :param coords: (Tensor), coordinates of voxels, (N, 4) (4 : Batch ind, x, y, z)
        :param values_in: (Tensor), features/tsdf, (N, C)
        :param inputs: dict: meta data from dataloader
        inputs['world_to_aligned_camera']: [bs, 4, 4]
        inputs['proj_matrices']: [bs, n_views, 3, 4, 4]
        :param scale:
        :param outputs:
        :param save_mesh: a bool to indicate whether or not to save the reconstructed mesh of current sample
        if direct_substitude:
        :return: outputs: dict: {
            'origin':                  (List), origin of the predicted partial volume,
                                    [3]
            'scene_tsdf':              (List), predicted tsdf volume,
                                    [(nx, ny, nz)]
            'target':                  (List), ground truth tsdf volume,
                                    [(nx', ny', nz')]
            'scene_name':                  (List), name of each scene in 'scene_tsdf',
                                    [string]
        }
        else:
        :return: updated_coords_all: (Tensor), updated coordinates, (N', 4) (4 : Batch ind, x, y, z)
        :return: values_all: (Tensor), features after gru fusion, (N', C)
        :return: tsdf_target_all: (Tensor), tsdf ground truth, (N', 1)
        :return: occ_target_all: (Tensor), occupancy ground truth, (N', 1)
        '''
        if self.global_volume[scale] is not None:
            # delete computational graph to save memory
            self.global_volume[scale] = self.global_volume[scale].detach()

        batch_size = len(inputs['fragment'])
        interval = 2 ** (self.n_layer - scale - 1)

        tsdf_target_all = None
        occ_target_all = None
        values_all = None
        updated_coords_all = None

        # ---incremental fusion----
        for i in range(batch_size):
            scene = inputs['scene'][i]  # scene name
            global_origin = inputs['vol_origin'][i]  # origin of global volume [3]
            origin = inputs['vol_origin_partial'][i]  # origin of part volume [3]

            if scene != self.scene_name[scale] and self.scene_name[scale] is not None and self.direct_substitude:
                outputs = self.save_mesh(scale, outputs, self.scene_name[scale])

            # if this fragment is from new scene, we reinitialize backend map
            if self.scene_name[scale] is None or scene != self.scene_name[scale]:
                self.scene_name[scale] = scene
                self.reset(scale)
                self.global_origin[scale] = global_origin

            # each level has its corresponding voxel size
            voxel_size = self.cfg.VOXEL_SIZE * interval

            # relative origin in global volume
            relative_origin = (origin - self.global_origin[scale]) / voxel_size # [3]
            relative_origin = relative_origin.cuda().long() # int grid

            batch_ind = torch.nonzero(coords[:, 0] == i).squeeze(1) # (number of voxels)
            if len(batch_ind) == 0:
                continue
            coords_b = torch.div(coords[batch_ind, 1:].long(), interval, rounding_mode='trunc') # int grid (number of voxels, 3) camera sys
            values = values_in[batch_ind] # (number of voxel, c) features/tsdf
            counts = counts_in[batch_ind] if self.enable_count else None # (number of voxel, 1) counts

            if 'occ_list' in inputs.keys():
                # get partial gt
                occ_target = inputs['occ_list'][self.n_layer - scale - 1][i] # [96//interval, 96//interval, 96//interval]
                tsdf_target = inputs['tsdf_list'][self.n_layer - scale - 1][i][occ_target] # (number of voxel)
                coords_target = torch.nonzero(occ_target) # (number of voxel, 3)
            else:
                coords_target = tsdf_target = None

            # convert to dense: 1. convert sparse feature to dense feature; 2. combine current feature coordinates and
            # previous feature coordinates within FBV from our backend map to get new feature coordinates (updated_coords)
            # updated_coords: (number of voxel, 3), current_volume: (H, W, Z, c), global_volume: (H, W, Z, c), 
            # target_volume: (H, W, Z, 1), valid: (number of voxel), valid_target: (number of voxel)
            updated_coords, current_volume, global_volume, target_volume, valid, valid_target, \
                current_counts_volume, global_counts_volume = self.convert2dense(
                coords_b,
                values,
                coords_target,
                tsdf_target,
                relative_origin,
                scale,
                counts)

            # dense to sparse: get features using new feature coordinates (updated_coords); 
            # 3D volume to 2D sparse voxels
            values = current_volume[updated_coords[:, 0], updated_coords[:, 1], updated_coords[:, 2]] # (number of voxel, c)
            global_values = global_volume[updated_coords[:, 0], updated_coords[:, 1], updated_coords[:, 2]] # (number of voxel, c)
            
            counts = current_counts_volume[updated_coords[:, 0], updated_coords[:, 1], updated_coords[:, 2]] if self.enable_count else None # (number of voxel, 1)
            global_counts = global_counts_volume[updated_coords[:, 0], updated_coords[:, 1], updated_coords[:, 2]] if self.enable_count else None # (number of voxel, 1)
            
            # get fused gt
            if target_volume is not None:
                tsdf_target = target_volume[updated_coords[:, 0], updated_coords[:, 1], updated_coords[:, 2]]
                occ_target = tsdf_target.abs() < 1
            else:
                tsdf_target = occ_target = None

            if self.enable_count:
                # update count
                total_counts = counts + global_counts
                values = values * (counts / (total_counts + 1e-9)) + global_values * (global_counts / (total_counts + 1e-9))
                counts = total_counts
            elif not self.direct_substitude:
                # convert to aligned camera coordinate; local coords -> aligned camera coords ???
                r_coords = updated_coords.detach().clone().float()
                r_coords = r_coords.permute(1, 0).contiguous().float() * voxel_size + origin.unsqueeze(-1).float()
                r_coords = torch.cat((r_coords, torch.ones_like(r_coords[:1])), dim=0)
                r_coords = inputs['world_to_aligned_camera'][i, :3, :] @ r_coords
                r_coords = torch.cat([r_coords, torch.zeros(1, r_coords.shape[-1]).to(r_coords.device)])
                r_coords = r_coords.permute(1, 0).contiguous()

                h = PointTensor(global_values, r_coords)
                x = PointTensor(values, r_coords)

                values = self.fusion_nets[scale](h, x) # (number of voxel, c)

            # feed back to global volume (direct substitute)
            self.update_map(values, updated_coords, target_volume, valid, valid_target, relative_origin, scale, counts)

            if updated_coords_all is None:
                updated_coords_all = torch.cat([torch.ones_like(updated_coords[:, :1]) * i, updated_coords * interval], # (number of voxel, 4)
                                               dim=1)
                values_all = values # (number of voxel, c)
                tsdf_target_all = tsdf_target # (number of voxel, 1)
                occ_target_all = occ_target # (number of voxel, 1)
            else:
                updated_coords = torch.cat([torch.ones_like(updated_coords[:, :1]) * i, updated_coords * interval],
                                           dim=1)
                updated_coords_all = torch.cat([updated_coords_all, updated_coords])
                values_all = torch.cat([values_all, values])
                if tsdf_target_all is not None:
                    tsdf_target_all = torch.cat([tsdf_target_all, tsdf_target])
                    occ_target_all = torch.cat([occ_target_all, occ_target])

            if self.direct_substitude and save_mesh:
                outputs = self.save_mesh(scale, outputs, self.scene_name[scale])

        if self.direct_substitude:
            return outputs
        else:
            return updated_coords_all, values_all, tsdf_target_all, occ_target_all
