import os
import numpy as np
import pickle
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset

from seg2d.mask2former.data.datasets.register_scannet_semantic import ids_to_nyu20
from tools.semantic_mapper import ids_to_scannet21


class ScanNetDataset(Dataset):
    def __init__(self, datapath, mode, transforms, nviews, n_scales, cfg):
        super(ScanNetDataset, self).__init__()
        self.datapath = datapath
        self.mode = mode
        self.n_views = nviews # 9
        self.transforms = transforms
        self.tsdf_file = 'all_tsdf_{}'.format(self.n_views)

        assert self.mode in ["train", "val", "test"]
        self.metas = self.build_list() # 19893 = scene x fragement; each scene: ~70 (scene0000_00) fragement x 9 views
        if mode == 'test':
            self.source_path = 'scans_test'
        else:
            self.source_path = 'scans'

        self.n_scales = n_scales # 2
        self.epoch = None
        self.tsdf_cashe = {}
        self.max_cashe = 100 # maximal
        self.W, self.H = cfg.WIDTH, cfg.HEIGHT
        
        # nerf
        self.enable_render = cfg.MODEL.ENABLE_RENDER
        self.enable_rgb = cfg.MODEL.NERF.ENABLE_RGB
        self.enable_sem = cfg.MODEL.NERF.ENABLE_SEM
        if self.enable_render:
            self.num_trg = cfg.MODEL.NERF.SAMPLE.NUM_TRG
            self.N_rays = cfg.MODEL.NERF.SAMPLE.N_RAYS
            self.render_stride = 1 if mode == 'trian' else cfg.MODEL.NERF.SAMPLE.RENDER_STRIDE
            
            if self.enable_sem: self.label_mapping = ids_to_nyu20() if cfg.MODEL.NUM_CLS == 20 else ids_to_scannet21()

    def build_list(self):
        with open(os.path.join(self.datapath, self.tsdf_file, 'fragments_{}.pkl'.format(self.mode)), 'rb') as f:
            metas = pickle.load(f)
        return metas

    def __len__(self):
        return len(self.metas)

    def read_cam_file(self, filepath, vid):
        intrinsics = np.loadtxt(os.path.join(filepath, 'intrinsic', 'intrinsic_color.txt'), delimiter=' ')[:3, :3]
        intrinsics = intrinsics.astype(np.float32)
        extrinsics = np.loadtxt(os.path.join(filepath, 'pose', '{}.txt'.format(str(vid))))
        return intrinsics, extrinsics

    def read_img(self, filepath):
        img = Image.open(filepath)
        return img

    def read_depth(self, filepath):
        # Read depth image and camera pose
        depth_im = cv2.imread(filepath, -1).astype(
            np.float32)
        depth_im /= 1000.  # depth is saved in 16-bit PNG in millimeters
        depth_im[depth_im > 3.0] = 0 # truncate
        return depth_im

    def read_sem(self, filepath):
        sem_im = cv2.imread(filepath, cv2.IMREAD_UNCHANGED).astype(np.int32)
        sem_im = self.label_mapping[sem_im].astype(np.int32)
        return sem_im
    

    def read_scene_volumes(self, data_path, scene):
        if scene not in self.tsdf_cashe.keys():
            if len(self.tsdf_cashe) > self.max_cashe:
                self.tsdf_cashe = {}
            full_tsdf_list = []
            for l in range(self.n_scales + 1): # multi-scale ground truth
                # load full tsdf volume
                full_tsdf = np.load(os.path.join(data_path, scene, 'full_tsdf_layer{}.npz'.format(l)),
                                    allow_pickle=True)
                full_tsdf_list.append(full_tsdf.f.arr_0)
            self.tsdf_cashe[scene] = full_tsdf_list
        return self.tsdf_cashe[scene]

    def __getitem__(self, idx):
        meta = self.metas[idx]

        imgs = []
        depth = []
        if self.enable_render and self.enable_sem:
            semantics = []
        extrinsics_list = []
        intrinsics_list = []

        tsdf_list = self.read_scene_volumes(os.path.join(self.datapath, self.tsdf_file), meta['scene'])

        for i, vid in enumerate(meta['image_ids']): # fragments data
            # load images
            imgs.append(
                self.read_img(
                    os.path.join(self.datapath, self.source_path, meta['scene'], 'color', '{}.jpg'.format(vid))))

            depth.append(
                self.read_depth(
                    os.path.join(self.datapath, self.source_path, meta['scene'], 'depth', '{}.png'.format(vid)))
            )
            
            if self.enable_render and self.enable_sem and self.mode != 'test':
                semantics.append(
                    self.read_sem(
                        os.path.join(self.datapath, self.source_path, meta['scene'], 'label-filt', '{}.png'.format(vid))
                    )
                )

            # load intrinsics and extrinsics
            intrinsics, extrinsics = self.read_cam_file(os.path.join(self.datapath, self.source_path, meta['scene']),
                                                        vid)

            intrinsics_list.append(intrinsics)
            extrinsics_list.append(extrinsics)

        intrinsics = np.stack(intrinsics_list)
        extrinsics = np.stack(extrinsics_list)

        items = {
            'imgs': imgs,
            'depth': depth,
            'intrinsics': intrinsics,
            'extrinsics': extrinsics,
            'tsdf_list_full': tsdf_list,
            'vol_origin': meta['vol_origin'],
            'scene': meta['scene'],
            'fragment': meta['scene'] + '_' + str(meta['fragment_id']),
            'epoch': [self.epoch],
        }
        
        if self.enable_sem and self.mode != 'test':
            items['semantics'] = semantics

        if self.transforms is not None:
            items = self.transforms(items)
        
        
        if self.enable_render:
            items = self.gen_rend_data(items)
        
        return items

    def gen_rend_data(self, items):
        idxs = torch.randint(len(items['imgs']), size=[self.num_trg])
        ret = {
            'rays': [],
            'trg_idxs': [],
        }
        if self.enable_rgb: ret['rgb'] = []
        if self.enable_sem and self.mode != 'test': ret['sem'] = []
        for idx in idxs:
            c2w = items['extrinsics'][idx]
            intr = items['intrinsics'][idx]
            rays = self.gen_rays(c2w, intr)
            ret['trg_idxs'].append(idx[None])
            
            if self.enable_rgb:
                rgb = items['imgs'][idx][:, ::self.render_stride, ::self.render_stride]
                rgb = rgb.reshape(3, -1).permute(1, 0) / 255.
            
            if self.enable_sem and self.mode != 'test':
                sem = items['semantics'][idx]
                sem = sem[::self.render_stride, ::self.render_stride]
                sem = sem.reshape(-1, 1) # N, 1
            
            if self.mode == 'train':
                ids = np.random.choice(len(rays), self.N_rays, replace=False)
                
                ret['rays'].append(rays[ids])
                if self.enable_rgb:
                    ret['rgb'].append(rgb[ids])
                
                if self.enable_sem:
                    ret['sem'].append(sem[ids])
                
            else:
                ret['rays'].append(rays)
                if self.enable_rgb:
                    ret['rgb'].append(rgb)
                if self.enable_sem and self.mode != 'test':
                    ret['sem'].append(sem)
                break
            
        for k in ret:
            ret[k] = torch.cat(ret[k], dim=0)
        
        items.pop('intrinsics')
        c2ws = items.pop('extrinsics')
        items['positions'] = c2ws[:, :3, 3]
        if 'semantics' in items: items.pop('semantics')
        
        items.update(ret)
        return items

    def gen_rays(self, c2w, intrinsic):
        H, W = self.H, self.W
        rays_o = c2w[:3, 3]
        X, Y = torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy')
        XYZ = torch.cat((X[:, :, None], Y[:, :, None], torch.ones_like(X[:, :, None])), dim=-1).float()
        XYZ = XYZ @ torch.linalg.inv(intrinsic).transpose(0, 1)
        XYZ = XYZ @ c2w[:3, :3].transpose(0, 1)
        rays_d = XYZ.view(-1, 3)

        rays = torch.cat([rays_o[None].repeat(H*W, 1), rays_d], dim=-1)
        rays = rays.view(H, W, -1)[::self.render_stride, ::self.render_stride].reshape(-1, 6)
        return rays

