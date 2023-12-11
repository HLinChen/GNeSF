import os
import torch
import trimesh
import numpy as np
import torchvision.utils as vutils
from skimage import measure
from loguru import logger
from tools.render import Visualizer
import cv2


from seg2d.mask2former.data.datasets.register_scannet_semantic import PALETTE


# print arguments
def print_args(args):
    logger.info("################################  args  ################################")
    for k, v in args.__dict__.items():
        logger.info("{0: <10}\t{1: <30}\t{2: <20}".format(k, str(v), str(type(v))))
    logger.info("########################################################################")


# torch.no_grad warpper for functions
def make_nograd_func(func):
    def wrapper(*f_args, **f_kwargs):
        with torch.no_grad():
            ret = func(*f_args, **f_kwargs)
        return ret

    return wrapper


# convert a function into recursive style to handle nested dict/list/tuple variables
def make_recursive_func(func):
    def wrapper(vars):
        if isinstance(vars, list):
            return [wrapper(x) for x in vars]
        elif isinstance(vars, tuple):
            return tuple([wrapper(x) for x in vars])
        elif isinstance(vars, dict):
            return {k: wrapper(v) for k, v in vars.items()}
        else:
            return func(vars)

    return wrapper


@make_recursive_func
def tensor2float(vars):
    if isinstance(vars, float):
        return vars
    elif isinstance(vars, torch.Tensor):
        if len(vars.shape) == 0:
            return vars.data.item()
        else:
            return [v.data.item() for v in vars]
    else:
        raise NotImplementedError("invalid input type {} for tensor2float".format(type(vars)))


@make_recursive_func
def tensor2numpy(vars):
    if isinstance(vars, np.ndarray):
        return vars
    elif isinstance(vars, torch.Tensor):
        return vars.detach().cpu().numpy().copy()
    else:
        raise NotImplementedError("invalid input type {} for tensor2numpy".format(type(vars)))


@make_recursive_func
def tocuda(vars):
    if isinstance(vars, torch.Tensor):
        return vars.cuda()
    elif isinstance(vars, str):
        return vars
    else:
        raise NotImplementedError("invalid input type {} for tensor2numpy".format(type(vars)))


def save_scalars(logger, mode, scalar_dict, global_step):
    scalar_dict = tensor2float(scalar_dict)
    for key, value in scalar_dict.items():
        if not isinstance(value, (list, tuple)):
            name = '{}/{}'.format(mode, key)
            logger.add_scalar(name, value, global_step)
        else:
            for idx in range(len(value)):
                name = '{}/{}_{}'.format(mode, key, idx)
                logger.add_scalar(name, value[idx], global_step)


def save_images(logger, mode, images_dict, global_step):
    images_dict = tensor2numpy(images_dict)

    def preprocess(name, img):
        if not (len(img.shape) == 3 or len(img.shape) == 4):
            raise NotImplementedError("invalid img shape {}:{} in save_images".format(name, img.shape))
        if len(img.shape) == 3:
            img = img[:, np.newaxis, :, :]
        img = torch.from_numpy(img[:1])
        return vutils.make_grid(img, padding=0, nrow=1, normalize=True, scale_each=True)

    for key, value in images_dict.items():
        if not isinstance(value, (list, tuple)):
            name = '{}/{}'.format(mode, key)
            logger.add_image(name, preprocess(name, value), global_step)
        else:
            for idx in range(len(value)):
                name = '{}/{}_{}'.format(mode, key, idx)
                logger.add_image(name, preprocess(name, value[idx]), global_step)


class DictAverageMeter(object):
    def __init__(self):
        self.data = {}
        self.count = 0

    def update(self, new_input):
        self.count += 1
        if len(self.data) == 0:
            for k, v in new_input.items():
                if not isinstance(v, float):
                    raise NotImplementedError("invalid data {}: {}".format(k, type(v)))
                self.data[k] = v
        else:
            for k, v in new_input.items():
                if not isinstance(v, float):
                    raise NotImplementedError("invalid data {}: {}".format(k, type(v)))
                self.data[k] += v

    def mean(self):
        return {k: v / self.count for k, v in self.data.items()}


def coordinates(voxel_dim, device=torch.device('cuda')):
    """ 3d meshgrid of given size.

    Args:
        voxel_dim: tuple of 3 ints (nx,ny,nz) specifying the size of the volume

    Returns:
        torch long tensor of size (3,nx*ny*nz)
    """

    nx, ny, nz = voxel_dim
    x = torch.arange(nx, dtype=torch.long, device=device)
    y = torch.arange(ny, dtype=torch.long, device=device)
    z = torch.arange(nz, dtype=torch.long, device=device)
    # x, y, z = torch.meshgrid(x, y, z)
    x, y, z = torch.meshgrid(x, y, z, indexing='ij') # ??? 
    return torch.stack((x.flatten(), y.flatten(), z.flatten()))


def apply_log_transform(tsdf):
    sgn = torch.sign(tsdf)
    out = torch.log(torch.abs(tsdf) + 1)
    out = sgn * out
    return out


def sparse_to_dense_torch_batch(locs, values, dim, default_val):
    dense = torch.full([dim[0], dim[1], dim[2], dim[3]], float(default_val), device=locs.device)
    dense[locs[:, 0], locs[:, 1], locs[:, 2], locs[:, 3]] = values
    return dense


def sparse_to_dense_torch(locs, values, dim, default_val, device):
    dense = torch.full([dim[0], dim[1], dim[2]], float(default_val), device=device, dtype=values.dtype)
    if locs.shape[0] > 0:
        dense[locs[:, 0], locs[:, 1], locs[:, 2]] = values
    return dense


def sparse_to_dense_channel(locs, values, dim, c, default_val, device):
    '''
    local coord
    convert 2d sparse voxel to 3d dense volume; 
    2D sparse volex to 3D volume:  -> (H, W, Z, c)
    locs: coordinates (N, 3)
    values: features (N, c)
    dim: list dimension [H, W, Z]
    c: channel int
    default_val: int default value
    
    return:
    dense: (H, W, Z, c)
    '''
    dense = torch.full([dim[0], dim[1], dim[2], c], float(default_val), device=device, dtype=values.dtype) # (H, W, Z, c)
    if locs.shape[0] > 0:
        dense[locs[:, 0], locs[:, 1], locs[:, 2]] = values
    return dense


def sparse_to_dense_np(locs, values, dim, default_val):
    dense = np.zeros([dim[0], dim[1], dim[2]], dtype=values.dtype)
    dense.fill(default_val)
    dense[locs[:, 0], locs[:, 1], locs[:, 2]] = values
    return dense


class SaveScene(object):
    def __init__(self, cfg):
        self.cfg = cfg
        # log_dir = cfg.LOGDIR.split('/')[-1]
        # self.log_dir = os.path.join('results', 'scene_' + cfg.DATASET + '_' + log_dir)
        log_dir = cfg.LOGDIR
        self.log_dir = os.path.join(log_dir, 'results')
        # print(self.log_dir)
        self.scene_name = None
        self.global_origin = None
        self.tsdf_volume = []  # not used during inference.
        self.weight_volume = []

        self.coords = None

        self.keyframe_id = None

        if cfg.VIS_INCREMENTAL:
            self.vis = Visualizer()

    def close(self):
        self.vis.close()
        cv2.destroyAllWindows()

    def reset(self):
        self.keyframe_id = 0
        self.tsdf_volume = []
        self.weight_volume = []

        # self.coords = coordinates(np.array([416, 416, 128])).float()

        # for scale in range(self.cfg.MODEL.N_LAYER):
        #     s = 2 ** (self.cfg.MODEL.N_LAYER - scale - 1)
        #     dim = tuple(np.array([416, 416, 128]) // s)
        #     self.tsdf_volume.append(torch.ones(dim).cuda())
        #     self.weight_volume.append(torch.zeros(dim).cuda())

    @staticmethod
    def tsdf2mesh(voxel_size, origin, tsdf_vol, rgb_vol=None, sem_vol=None, attribute='semseg'): # 
        verts, faces, norms, vals = measure.marching_cubes(tsdf_vol, level=0)
        verts_ind = np.round(verts).astype(int)
        
        verts = verts * voxel_size + origin  # voxel grid coordinates to world coordinates
        
        vertex_attributes = {}
        # get vertex attributes
        if sem_vol is not None:
            semseg = sem_vol[verts_ind[:,0],verts_ind[:,1],verts_ind[:,2]]
            vertex_attributes['semseg'] = semseg
        
        # color mesh
        if attribute=='color':
            # colors = rgb_vol[:, verts_ind[:,0],verts_ind[:,1],verts_ind[:,2]].T
            colors = rgb_vol[verts_ind[:,0],verts_ind[:,1],verts_ind[:,2], :]
        elif attribute=='semseg':
            cmap = np.array(PALETTE) # FIXME: support more general colormaps
            label_viz = semseg.copy()
            # label_viz[(label_viz<0) | (label_viz>=len(cmap))] = len(cmap)
            label_viz[label_viz == 255] = len(cmap)-1
            # colors = cmap[label_viz, :]
            colors = cmap[label_viz]
        else:
            colors = None
        
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=norms, \
                    vertex_attributes=vertex_attributes, vertex_colors=colors)
        return mesh

    def vis_incremental(self, epoch_idx, batch_idx, imgs, outputs):
        tsdf_volume = outputs['scene_tsdf'][batch_idx].data.cpu().numpy()
        origin = outputs['origin'][batch_idx].data.cpu().numpy()
        if self.cfg.DATASET == 'demo':
            origin[2] -= 1.5

        if (tsdf_volume == 1).all():
            logger.warning('No valid partial data for scene {}'.format(self.scene_name))
        else:
            # Marching cubes
            mesh = self.tsdf2mesh(self.cfg.MODEL.VOXEL_SIZE, origin, tsdf_volume)
            # vis
            key_frames = []
            for img in imgs[::3]:
                img = img.permute(1, 2, 0)
                img = img[:, :, [2, 1, 0]]
                img = img.data.cpu().numpy()
                img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
                key_frames.append(img)
            key_frames = np.concatenate(key_frames, axis=0)
            cv2.imshow('Selected Keyframes', key_frames / 255)
            cv2.waitKey(1)
            # vis mesh
            self.vis.vis_mesh(mesh)

    def save_incremental(self, epoch_idx, batch_idx, imgs, outputs):
        save_path = os.path.join('incremental_' + self.log_dir + '_' + str(epoch_idx), self.scene_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        tsdf_volume = outputs['scene_tsdf'][batch_idx].data.cpu().numpy()
        origin = outputs['origin'][batch_idx].data.cpu().numpy()
        if self.cfg.DATASET == 'demo':
            origin[2] -= 1.5

        if (tsdf_volume == 1).all():
            logger.warning('No valid partial data for scene {}'.format(self.scene_name))
        else:
            # Marching cubes
            mesh = self.tsdf2mesh(self.cfg.MODEL.VOXEL_SIZE, origin, tsdf_volume)
            # save
            mesh.export(os.path.join(save_path, 'mesh_{}.ply'.format(self.keyframe_id)))

    def save_scene_eval(self, epoch, outputs, batch_idx=0):
        tsdf_volume = outputs['scene_tsdf'][batch_idx].data.cpu().numpy()
        origin = outputs['origin'][batch_idx].data.cpu().numpy()

        if (tsdf_volume == 1).all():
            logger.warning('No valid data for scene {}'.format(self.scene_name))
        else:
            # save tsdf volume for atlas evaluation
            data = {'origin': origin,
                    'voxel_size': self.cfg.MODEL.VOXEL_SIZE,
                    'tsdf': tsdf_volume}
            if 'scene_rgb' in outputs:
                rgb_volume = outputs['scene_rgb'][batch_idx].data.cpu().numpy() # .permute(3, 0, 1, 2)
                data['rgb'] = rgb_volume
            else:
                rgb_volume = None
                
            if 'scene_sem' in outputs:
                sem_volume = outputs['scene_sem'][batch_idx].data.cpu().numpy()
                data['sem'] = sem_volume
            else:
                sem_volume = None
                
            # Marching cubes
            mesh = self.tsdf2mesh(self.cfg.MODEL.VOXEL_SIZE, origin, tsdf_volume, rgb_volume, sem_volume)
                
            save_path = '{}_fusion_eval_{}'.format(self.log_dir, epoch)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            np.savez_compressed(
                os.path.join(save_path, '{}.npz'.format(self.scene_name)),
                **data)
            mesh.export(os.path.join(save_path, '{}.ply'.format(self.scene_name)))

    def __call__(self, outputs, inputs, epoch_idx):
        # no scene saved, skip
        if "scene_name" not in outputs.keys():
            return

        batch_size = len(outputs['scene_name'])
        for i in range(batch_size):
            scene = outputs['scene_name'][i]
            self.scene_name = scene.replace('/', '-')

            if self.cfg.SAVE_SCENE_MESH:
                self.save_scene_eval(epoch_idx, outputs, i)


class ImageWriter:
    """ Saves image to logdir during training"""

    def __init__(self, save_path, mode='train', w=640, h=480):
        # self._save_path = os.path.join(save_path, "viz", mode, "image")
        self._save_path = os.path.join(save_path, "image")
        os.makedirs(self._save_path, exist_ok=True)
        # _rgb_path = os.path.join(self._save_path, "rgb")
        # _depth_path = os.path.join(self._save_path, "depth")
        # os.makedirs(_rgb_path, exist_ok=True)
        # os.makedirs(_depth_path, exist_ok=True)
        self.w = w
        self.h = h
        self.cmap = np.array(PALETTE) # np.array(NYU40_COLORMAP[1:])

    def save_image(self, batch, out, names, render_stride=1):
        bs = batch['rgb'].shape[0]
        h = self.h // render_stride
        w = self.w // render_stride
        res = {}
        for k, pre_img in out.items():
            if k in ['rgb', 'depth', 'sem']:
                c = pre_img.shape[-1] if len(pre_img.shape) > 2 else 1
                
                if k != 'sem': pre_img = pre_img.view(-1, h, w, c)
                
                if k == 'rgb':
                    gt_img = batch[k]
                    gt_img = gt_img.view(-1, h, w, c)
                    gt_img = (gt_img * 255).clip(0, 255).to(torch.uint8)
                    
                    pre_img = (pre_img * 255).clip(0, 255).to(torch.uint8).cpu()
                    
                    img = torch.cat([gt_img, pre_img], dim=-2)
                    img = img.flip(-1).detach().numpy()
                    # img = img.astype(np.uint8)
                elif k == 'depth':
                    imgs = []
                    for i in range(bs):
                        # pre_img[i] = (pre_img[i] - pre_img[i].min()) / (pre_img[i].max() - pre_img[i].min())
                        img, [mi, ma] = visualize_depth_numpy(pre_img[i].detach().cpu().numpy())
                        imgs.append(img)
                    img = np.stack(imgs, axis=0) # .flip(img, -1)
                    img = np.flip(img, -1)
                elif k == 'sem':
                    gt_sem = batch[k].cpu().numpy().copy() # .reshape(h, w)
                    # gt_sem = gt_sem.reshape(bs, h, w, -1)
                    
                    pre_sem = pre_img.argmax(dim=-1, keepdim=True).cpu().numpy() # .reshape(h, w)
                    pre_sem[gt_sem==255] = self.cmap.shape[0]-1
                    pre_sem = self.cmap[pre_sem].reshape(bs, h, w, 3)
                    
                    gt_sem[gt_sem==255] = self.cmap.shape[0]-1
                    gt_sem = self.cmap[gt_sem].reshape(bs, h, w, 3)
                    
                    img = np.concatenate([gt_sem, pre_sem], axis=-2)
            
                # # img = cv2.colorChange(img, cv2.COLOR_RGB2BGR)
                # cv2.imwrite(os.path.join(self._save_path, k, name), img)
                res[k] = img
                
        img = np.concatenate([res[k] for k in res.keys()], axis=-2)
        for i in range(bs):
            cv2.imwrite(os.path.join(self._save_path, names[i]+"_{}.jpg".format(i)), img[i])
        
        return


def visualize_depth_numpy(depth, minmax=None, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """

    x = np.nan_to_num(depth)  # change nan to 0
    if minmax is None:
        mi = np.min(x[x > 0])  # get minimum positive depth (ignore background)
        ma = np.max(x)
    else:
        mi, ma = minmax

    x = (x - mi) / (ma - mi + 1e-8)  # normalize to 0~1
    x = (255 * x).astype(np.uint8)
    x_ = cv2.applyColorMap(x, cmap)
    return x_, [mi, ma]


def info_print(epoch_idx, avg_scalars, mode='Train'):
    info = '{} Epoch[{}] '.format(mode, epoch_idx)
    for k,v in avg_scalars.items():
        info += '{}: {:.2f} '.format(k.replace('_occ', ''), v)
    logger.info("{}".format(info))


def compute_psnr(x, y, mask=None):
    if mask == None:
        psnr = 20.0 * torch.log10(
                    1.0 / (((x - y) ** 2).mean() / (3.0)).sqrt())
    else:
        psnr = 20.0 * torch.log10(
                    1.0 / (((x[mask] - y[mask]) ** 2).mean() / (3.0)).sqrt())
    
    return psnr.cpu()
