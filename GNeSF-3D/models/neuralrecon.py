import torch
import torch.nn as nn

from .backbone import Encoderdecoder2D
from .neucon_network import NeuConNet
from .gru_fusion import GRUFusion
from utils import tocuda, compute_psnr


class NeuralRecon(nn.Module):
    '''
    NeuralRecon main class.
    '''

    def __init__(self, cfg):
        super(NeuralRecon, self).__init__()
        self.cfg = cfg.MODEL
        # other hparams
        self.n_scales = len(self.cfg.THRESHOLDS) - 1 # 2
        
        self.backbone2d = Encoderdecoder2D(cfg.MODEL)
        self.neucon_net = NeuConNet(cfg.MODEL) # 3D FPN
        # for fusing to global volume
        ch_sem = 1 if self.cfg.NERF.ENABLE_SEM else 0
        ch_rgb = 3 if self.cfg.NERF.ENABLE_RGB else 0
        ch_in = [1+ch_sem+ch_rgb] * 3 if self.cfg.ENABLE_RENDER else None
        self.fuse_to_global = GRUFusion(cfg.MODEL, ch_in, direct_substitute=True)

    def forward(self, inputs, save_mesh=False):
        '''
        each of a batch = a fragments (local/partial)
        :param inputs: dict: {
            'imgs':                    (Tensor), images,
                                    (batch size, number of views, C, H, W)
            'vol_origin':              (Tensor), origin of the full voxel volume (xyz position of voxel (0, 0, 0)),
                                    (batch size, 3)
            'vol_origin_partial':      (Tensor), origin of the partial voxel volume (xyz position of voxel (0, 0, 0)),
                                    (batch size, 3)
            'world_to_aligned_camera': (Tensor), matrices: transform from world coords to aligned camera coords,
                                    (batch size, number of views, 4, 4)
            'proj_matrices':           (Tensor), projection matrix,
                                    (batch size, number of views, number of scales, 4, 4)
            when we have ground truth:
            'tsdf_list':               (List), tsdf ground truth for each level,
                                    [(batch size, DIM_X, DIM_Y, DIM_Z)]
            'occ_list':                (List), occupancy ground truth for each level,
                                    [(batch size, DIM_X, DIM_Y, DIM_Z)]
            others: unused in network
        }
        :param save_mesh: a bool to indicate whether or not to save the reconstructed mesh of current sample
        :return: outputs: dict: {
            'coords':                  (Tensor), coordinates of voxels,
                                    (number of voxels, 4) (4 : batch ind, x, y, z)
            'tsdf':                    (Tensor), TSDF of voxels,
                                    (number of voxels, 1)
            When it comes to save results:
            'origin':                  (List), origin of the predicted partial volume,
                                    [3]
            'scene_tsdf':              (List), predicted tsdf volume,
                                    [(nx, ny, nz)]
        }
                 loss_dict: dict: {
            'tsdf_occ_loss_X':         (Tensor), multi level loss
            'total_loss':              (Tensor), total loss
        }
        '''
        inputs = tocuda(inputs)
        outputs = {}

        # # image feature extraction
        # # in: images; out: feature maps
        out = self.backbone2d(inputs['imgs'])

        outputs, loss_dict = self.neucon_net(out['features'], inputs, outputs, out['sem_seg'])

        # fuse to global volume. tsdf
        if not self.training and 'coords' in outputs.keys():
            outputs = self.fuse_to_global(outputs['coords'], outputs['tsdf'], inputs, self.n_scales, outputs, save_mesh)
                
        # gather loss.
        print_loss = 'Loss: '
        for k, v in loss_dict.items():
            print_loss += f'{k}: {v} '

        weighted_loss = 0 if len(loss_dict) > 0 else torch.tensor(0.0).cuda()

        for i, (k, v) in enumerate(loss_dict.items()):
            weighted_loss += v * self.cfg.LW[i]

        if self.cfg.ENABLE_RENDER and len(outputs) > 0:
            with torch.no_grad():
                psnr = compute_psnr(outputs['rgb'], inputs['rgb'])
                loss_dict['psnr'] = psnr
                
        loss_dict.update({'total_loss': weighted_loss})
        return outputs, loss_dict
