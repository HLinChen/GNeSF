import os
import numpy as np
import imageio
import torch
from torch.utils.data import Dataset
import sys
import cv2
import threading, queue
import tqdm
import json
import detectron2.data.transforms as T

sys.path.append('../')
from .data_utils import get_nearest_pose_ids, random_flip
from .scannet_data_utils import scannet20_colour_code

from seg2d.mask2former.data.datasets.register_scannet_semantic import ids_to_nyu20


class ScanNetDataset(Dataset):
    def __init__(self, args, mode, scenes=(), **kwargs):
        base_dir = os.path.join(args.datadir, 'scannet/scans')
        self.args = args
        self.mode = mode  # train / test / validation
        self.num_source_views = args.num_source_views
        self.depth_range = [0.1, 3]
        self.H, self.W = args.image_size
        self.crop = args.crop # 6
        self.pad = args.pad # 2
        self.scenes = scenes
        
        if args.enable_semantic or self.args.enable_2dsem_seg:
            if self.args.num_cls == 20:
                self.label_mapping = ids_to_nyu20(os.path.join(args.datadir, 'scannet')) # load_scannet_nyu40_mapping(os.path.join(args.datadir, 'scannet'))
                self.colour_map = torch.from_numpy(scannet20_colour_code)
        else:
            self.colour_map = None
        
        self.build(base_dir, mode)
        
        # preload dataset
        if args.preload:
            self.imgs = self.preload_threading(self.get_image)
            
            if args.enable_semantic:
                self.semantics = self.preload_threading(
                    self.get_semantic, data_str="semantics")

    def __len__(self):
        return len(self.render_ids)

    def __getitem__(self, idx):
        render_id_in_total = self.render_ids[idx] # number
        rgb = self.imgs[render_id_in_total] if self.args.preload else \
            self.get_image(render_id_in_total)
        render_pose = self.poses_total[render_id_in_total]
        rgb_file = self.rgb_files_total[render_id_in_total]
        
        if self.args.enable_semantic:
            sem = self.semantics[render_id_in_total] if self.args.preload else \
                self.get_semantic(render_id_in_total)
            sem_file = self.sem_files_total[render_id_in_total]
        else:
            sem = None
        
        rgb = rgb.astype(np.float32) / 255.
        
        depth = None

        depth_range = torch.tensor(self.depth_range)

        scene_id = self.render_scene_ids[idx]
        intr = self.intrinsics_total[scene_id]
        train_ids_in_total = self.train_ids[scene_id] # [., ., ]
        train_poses = self.poses_total[train_ids_in_total]

        img_size = rgb.shape[:2]
        camera = np.concatenate((list(img_size), intr.flatten(),
                                 render_pose.flatten())).astype(np.float32)

        if self.mode == 'train':
            render_id_in_train =  train_ids_in_total.index(render_id_in_total)
            subsample_factor = np.random.choice(np.arange(1, 4), p=[0.2, 0.45, 0.35])
            num_select = self.num_source_views + np.random.randint(low=-2, high=3)
        else:
            render_id_in_train = -1
            subsample_factor = 1
            num_select = self.num_source_views

        nearest_pose_ids = get_nearest_pose_ids(render_pose,
                                                train_poses,
                                                min(self.num_source_views*subsample_factor, 20),
                                                tar_id=render_id_in_train,
                                                angular_dist_method='dist')
        nearest_pose_ids = np.random.choice(nearest_pose_ids, min(num_select, len(nearest_pose_ids)), replace=False)

        assert render_id_in_train not in nearest_pose_ids
        # occasionally include input image
        if np.random.choice([0, 1], p=[0.995, 0.005]) and self.mode == 'train':
            nearest_pose_ids[np.random.choice(len(nearest_pose_ids))] = render_id_in_train

        src_rgbs = []
        src_sems = []
        src_sem_paths = []
        src_cameras = []
        for id in nearest_pose_ids:
            train_id_in_total = train_ids_in_total[id]
            src_rgb = self.imgs[train_id_in_total] if self.args.preload else \
                self.get_image(train_id_in_total)
            
            if self.args.enable_2dsem_seg:
                src_sem = self.semantics[train_id_in_total] if self.args.preload else \
                    self.get_semantic(train_id_in_total)
                src_sem_paths.append(self.sem_files_total[train_id_in_total])
                
                
            if self.args.enable_2dsem_seg:
                src_sems.append(src_sem)
                
            src_rgb = src_rgb.astype(np.float32) / 255.
            src_rgbs.append(src_rgb)
            train_pose = train_poses[id]
            img_size = src_rgb.shape[:2]
            src_camera = np.concatenate((list(img_size), intr.flatten(),
                                              train_pose.flatten())).astype(np.float32)
            src_cameras.append(src_camera)

        src_rgbs = np.stack(src_rgbs, axis=0)
        src_cameras = np.stack(src_cameras, axis=0)
        if self.args.enable_2dsem_seg:
            src_sems = np.stack(src_sems, axis=0)
        else:
            src_sems = None


        if self.mode == 'train' and np.random.choice([0, 1]):
            out = random_flip(rgb, camera, src_rgbs, src_cameras, sem, depth, src_sems)
            rgb, camera, src_rgbs, src_cameras = out['rgb'], out['camera'], out['src_rgbs'], out['src_cameras']
            
            if self.args.enable_semantic: sem = out['sem']
            if self.args.enable_2dsem_seg: src_sems = out['src_sems']


        sample = {'rgb': torch.from_numpy(rgb[..., :3]),
                'camera': torch.from_numpy(camera),
                'rgb_path': rgb_file,
                'src_rgbs': torch.from_numpy(src_rgbs[..., :3]),
                'src_cameras': torch.from_numpy(src_cameras),
                'depth_range': depth_range,
                }

        if self.args.enable_semantic:
            sem = torch.from_numpy(sem.copy()).long()
            sem[sem==255] = -1
            sample['sem'] = sem # - 1 # shift void class from value 0 to -1, to match self.ignore_label
            sample['sem_path'] = sem_file
        if self.args.enable_2dsem_seg:
            sample['src_sems'] = torch.from_numpy(src_sems.copy()).long() # - 1 # shift void class from value 0 to -1, to match self.ignore_label
            sample['src_sem_paths'] = src_sem_paths
        return sample

    def get_scene(self, dir='./data'):
        file_path = '{}/scannetv2_{}.txt'.format(dir, self.mode.replace('_eval', ''))
        
        with open(file_path) as f:
            scenes = f.readlines()
        
        scenes = [i.strip() for i in scenes]
        
        return scenes

    def get_intr(self, dir, scene):
        with open(os.path.join(dir, "{}.txt".format(scene))) as info_f:
            info = [line.rstrip().split(' = ') for line in info_f]
            info = {key:value for key, value in info}
            H, W = float(info['colorHeight']), float(info['colorWidth'])
            intrinsics = np.array([
                [float(info['fx_color']), 0, float(info['mx_color']), 0],
                [0, float(info['fy_color']), float(info['my_color']), 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]], dtype=np.float32)
        
        intrinsics = self.process_intr(intrinsics, H, W)
        
        return intrinsics
    
    def process_intr(self, intrinsics , raw_H, raw_W): #
        # resize
        if raw_H == 968 and raw_W == 1296 and self.pad > 0:
            intrinsics[1, 2] += self.pad
            raw_H += self.pad * 2
        scale_x = self.W / raw_W
        scale_y = self.H / raw_H
        intrinsics[0, :] *= scale_x
        intrinsics[1, :] *= scale_y
        if self.crop > 0:
            intrinsics[0, 2] -= self.crop
            intrinsics[1, 2] -= self.crop
        return intrinsics
    
    def get_image(self, idx):
        image_path = self.rgb_files_total[idx]
        # directly using PIL.Image.open() leads to weird corruption....
        image = imageio.imread(image_path, as_gray=False, pilmode="RGB")
        # image = imageio.imread(image_path, mode="RGB")
        
        image = self.preprocess_image(image)
        return image
    
    def get_semantic(self, idx):
        # Read semantic label and remap
        image_path = self.sem_files_total[idx]
        semantic_im = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        semantic_im = self.preprocess_semantic(semantic_im)
        return semantic_im
    
    def get_depth(self, idx):
        # Read depth map
        depth_path = self.depth_files_total[idx]
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED) / 1000.0
        depth = self.preprocess_depth(depth)
        return depth
    
    def preprocess_image(self, image):
        # pad 4 pixels to height so that images have aspect ratio of 4:3
        h, w = image.shape[:2]
        if h == 968 and w == 1296 and self.pad > 0:
            image = cv2.copyMakeBorder(src=image, top=self.pad, bottom=self.pad, left=0, right=0, borderType=cv2.BORDER_CONSTANT, value=[0,0,0])
        
        self.raw_H, self.raw_W = image.shape[:2]

        # resize
        image = cv2.resize(image, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        if self.crop > 0:
            image = image[self.crop: -self.crop, self.crop: -self.crop, :]
        return image

    def preprocess_semantic(self, semantic_im):
        h, w = semantic_im.shape[:2]
        if h == 968 and w == 1296 and self.pad > 0:
            semantic_im = cv2.copyMakeBorder(src=semantic_im, top=self.pad, bottom=self.pad, left=0, right=0, borderType=cv2.BORDER_CONSTANT, value=0)
        semantic_im = cv2.resize(semantic_im, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
        if self.crop > 0:
            semantic_im = semantic_im[self.crop: -self.crop, self.crop: -self.crop]
        semantic_im_map = semantic_im.copy()
        
        semantic_im_map = self.label_mapping[semantic_im]
        
        semantic_im_map = semantic_im_map.astype(np.uint8) # .astype(np.int16)
        return semantic_im_map
    
    def preprocess_depth(self, depth):
        h, w = depth.shape[:2]
        if self.crop > 0:
            depth = depth[self.crop: -self.crop, self.crop: -self.crop]

        if h != self.H or w != self.W:
            # resize
            depth = cv2.resize(depth, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        return depth
    
    def preload_worker(self,data_list, load_func, q,lock, idx_tqdm):
        while True:
            idx = q.get()
            data_list[idx] = load_func(idx)
            with lock:
                idx_tqdm.update()
            q.task_done()

    def preload_threading(self, load_func, data_str="images"):
        data_list = [None]*len(self.rgb_files_total)
        q = queue.Queue(maxsize=len(self.rgb_files_total))
        idx_tqdm = tqdm.tqdm(range(len(self.rgb_files_total)),desc="preloading {}".format(data_str),leave=False)
        for i in range(len(self.rgb_files_total)): q.put(i)
        lock = threading.Lock()
        for ti in range(self.args.workers): # *4
            t = threading.Thread(target=self.preload_worker,
                                 args=(data_list,load_func,q,lock,idx_tqdm),daemon=True)
            t.start()
        q.join()
        idx_tqdm.close()
        assert(all(map(lambda x: x is not None,data_list)))
        return data_list

    # def load_data(self, scene_dir, scene, split_dir=None):
    def load_data(self, scene_dir, scene, cam_info=None):
        total_ids = os.listdir(os.path.join(scene_dir, 'color'))
        total_ids = sorted([int(os.path.splitext(frame)[0]) for frame in total_ids])
        if self.mode == 'train':
            sample_ids = total_ids[::self.args.sample_step]
        else:
            sample_ids = total_ids[self.args.sample_step // 2 :: self.args.sample_step]
        
        if cam_info is None:
            intr = self.get_intr(scene_dir, scene)
        else:
            intrinsic_all = cam_info['intrinsics']
            pose_all = cam_info['poses']
            
            intr, raw_H, raw_W = intrinsic_all[scene]
            intr = self.process_intr(intr, int(raw_H), int(raw_W))
            
            poses = pose_all[scene]
        
        img_dir_list, pose_list = [], []
        semantic_dir_list = []
        depth_dir_list = []
        pose_list = []
        id_list = [i for i in sample_ids if i in poses]
        for i, frame_id in enumerate(sample_ids):
            # camera to world c2w
            if cam_info is None:
                pose = np.loadtxt(os.path.join(scene_dir, 'pose', '%d.txt' % frame_id)) # [:3]
            else:
                if frame_id not in poses: continue
                pose = np.array(poses[frame_id]).astype(np.float32)

            # skip frames with no valid pose
            if not np.all(np.isfinite(pose)):
                continue

            img_dir_list.append(os.path.join(scene_dir, 'color', '%d.jpg'%frame_id))
            pose_list.append(pose)
            # id_list.append(frame_id)
            
            depth_dir_list.append(os.path.join(scene_dir, 'depth', '%d.png'%frame_id))
            
            semantic_dir_list.append(os.path.join(scene_dir, 'label-filt', '%d.png'%frame_id))
            

        poses = np.stack(pose_list, axis=0)
        
        out = {
            'rgb_files':  img_dir_list,
            'poses':    poses,
            'intr':     intr,
        }
        out['frame_id'] = id_list
        
        if self.args.enable_semantic or self.args.enable_2dsem_seg:
            out['sem_files'] = semantic_dir_list
        return out

    def build(self, base_dir, mode):
        self.poses_total = [] # all
        self.intrinsics_total = [] # same in the same scene
        self.rgb_files_total = [] # all
        self.sem_files_total = [] # all
        self.depth_files_total = [] # all
        
        self.train_ids = [] # id of train in all data [[scene], []]
        self.render_ids = [] # id of render in all data [, , ]
        self.render_scene_ids = [] # id of render in train [0, 0, 1, 1, ]
        

        split_dir = './data'
        scenes = self.get_scene(split_dir) if len(self.scenes) == 0 else self.scenes
        out_eval_data = True # True and mode == 'train' # True False
        if out_eval_data:
            eval_data = {}
            if os.path.exists('data/scannet_ids.json'):
                with open('data/scannet_ids.json', 'r') as f:
                    eval_data = json.load(f)
        
        
        cam_info_all = torch.load(os.path.join(self.args.datadir, 'scannet', 'cam_info_all.pth'))
        cam_info = cam_info_all['camera_info']
        poses_list = cam_info_all['poses_list']
        
        for scene_name, scene in cam_info['poses'].items():
            for cam_index, poses in scene.items():
                index = scene[cam_index]
                scene[cam_index] = poses_list[index, ...]

        for i, scene in enumerate(tqdm.tqdm(scenes)):
            scene_dir = os.path.join(base_dir, scene)
            id_end = len(self.poses_total)
            data = self.load_data(scene_dir, scene, cam_info) # pose c2w
            
            rgb_files, poses, intr = data['rgb_files'], data['poses'], data['intr']

            self.poses_total.extend(poses)
            self.intrinsics_total.append(intr)
            self.rgb_files_total.extend(rgb_files)
            
            if self.args.enable_semantic or self.args.enable_2dsem_seg:
                sem_files = data['sem_files']
                self.sem_files_total.extend(sem_files)
                
            
            if mode == 'train':
                i_train = np.array(np.arange(int(poses.shape[0])))
                i_render = i_train
                if out_eval_data:
                    eval_data[scene] = data['frame_id'] # [::self.args.llffhold*5]
            else:
                if self.args.llffhold > 1:
                    step = self.args.llffhold if mode == 'val' else self.args.llffhold * 5
                    i_test = np.arange(poses.shape[0])[::step]
                    i_train = np.array([j for j in np.arange(int(poses.shape[0])) if
                                        (j not in i_test and j not in i_test)])
                    i_render = i_test
                    if out_eval_data:
                        # eval_data[scene] = data['frame_id'] # [::step]
                        eval_data[scene] = data['frame_id'] if scene not in eval_data else sorted(list(set(data['frame_id']) or set(eval_data[scene])))
                else:
                    i_train = np.array(np.arange(int(poses.shape[0])))
                    i_render = i_train
            
            
            self.train_ids.append([id_end + i for i in i_train])
            self.render_ids.extend([id_end + i for i in i_render])
            num_render = len(i_render)
            self.render_scene_ids.extend([i]*num_render)

        assert len(self.render_ids) == len(set(self.render_ids))
        self.poses_total = np.array(self.poses_total)
        
        assert len(self.render_ids) == len(set(self.render_ids))
        self.poses_total = np.array(self.poses_total)
        
        if out_eval_data:
            with open('data/scannet_ids.json', 'w') as f:
                json.dump(eval_data, f)

        return