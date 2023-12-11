# This file is derived from [Atlas](https://github.com/magicleap/Atlas).
# Originating Author: Zak Murez (zak.murez.com)
# Modified for [NeuralRecon](https://github.com/zju3dv/NeuralRecon) by Yiming Xie.

# Original header:
# Copyright 2020 Magic Leap, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import os
from skimage import measure

sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.append('.')
sys.path.append('..')
import argparse
import json
os.environ['PYOPENGL_PLATFORM'] = 'egl'
# # from pyrender.platforms import egl

# try:
#     import pyrender
# except ImportError:
#     pass

# import pyrender
import numpy as np
import torch
import trimesh
from tqdm import tqdm

from tools.simple_loader import *
from tools.evaluation_utils import project_to_mesh, compute_miou
import open3d as o3d
import ray
import plyfile

from tools.semantic_mapper import nyu40_to_scannet21
torch.multiprocessing.set_sharing_strategy('file_system')


def parse_args():
    parser = argparse.ArgumentParser(description="NeuralRecon ScanNet Testing")
    parser.add_argument("--model", required=True, metavar="FILE",
                        help="path to checkpoint")
    parser.add_argument('--max_depth', default=10., type=float,
                        help='mask out large depth values since they are noisy')
    parser.add_argument("--data_path", metavar="DIR",
                        help="path to dataset", default='/data/hlchen/data/scannet/scans_test')
    parser.add_argument("--gt_path", metavar="DIR",
                        help="path to raw dataset", default='/data/hlchen/data/scannet/scans_test')
    parser.add_argument("--mode",
                        help="path to raw dataset", default='val')
    parser.add_argument('--num_cls', type=int, default=20, help='#number of gpus')
    parser.add_argument("--dataset", help="path to raw dataset", default='scannet')

    # ray config
    parser.add_argument('--n_proc', type=int, default=2, help='#processes launched to process scenes.')
    parser.add_argument('--n_gpu', type=int, default=1, help='#number of gpus')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--loader_num_workers', type=int, default=8)
    return parser.parse_args()


args = parse_args()

if args.dataset == 'scannet':
    # remapper = ids_to_nyu20()
    remapper = np.ones(150) * (255)
    main_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
    for i, x in enumerate(main_ids):
        remapper[x] = i

    inv_remapper = np.ones(21)
    for i, x in enumerate(main_ids):
        inv_remapper[i] = x
elif args.dataset == 'scannet21':
    remapper = np.zeros(150)
    mapping = nyu40_to_scannet21()
    for k, v in mapping.items():
        remapper[k] = v
        
    remapper = remapper - 1
    remapper[remapper==-1] = 255
    

def read_label(fn, is_gt=False):
    a = plyfile.PlyData().read(fn)
    w = np.array(a.elements[0]['label'])
    if is_gt: w = remapper[w] if remapper is not None else w
    
    w = w.astype(np.int64)
    
    return w


def process(scene, total_scenes_index, total_scenes_count, mode='val'):

    save_path = os.path.join(args.model, mode)
    os.makedirs(save_path, exist_ok=True)
    width, height = 640, 480
    
    if args.dataset == 'scannet':
        intrinsic_dir = os.path.join(args.data_path, scene, 'intrinsic', 'intrinsic_depth.txt')
        cam_intr = np.loadtxt(intrinsic_dir, delimiter=' ')[:3, :3]
        test_framid = os.listdir(os.path.join(args.data_path, scene, 'color'))
        n_imgs = len(test_framid)
        dataset = ScanNetDataset(n_imgs, scene, args.data_path, args.max_depth)
    voxel_size = 4

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=float(voxel_size) / 100,
        sdf_trunc=3 * float(voxel_size) / 100,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
        # color_type=o3d.integration.TSDFVolumeColorType.RGB8)

    # transfer labels from pred mesh to gt mesh using nearest neighbors
    # file_attributes = os.path.join(save_path, '%s_attributes.npz'%scene)
    file_attributes = os.path.join(args.model, '%s.npz'%scene)
    if args.dataset == 'scannet':
        ply_postfix = '_vh_clean_2.labels.ply' if mode != 'test' else '_vh_clean_2.ply'
        file_mesh_trgt = os.path.join(args.gt_path, scene, scene + ply_postfix)
        
    item = np.load(file_attributes)
    
    verts, faces, norms, vals = measure.marching_cubes(item['tsdf'], level=0)
    verts_ind = np.round(verts).astype(int)
    
    verts = verts * item['voxel_size'] + item['origin']
    
    semseg = item['sem'][verts_ind[:,0],verts_ind[:,1],verts_ind[:,2]]
    vertex_attributes = {}
    vertex_attributes['semseg'] = semseg
    
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=norms, \
                vertex_attributes=vertex_attributes)
    
    mesh_trgt = trimesh.load(file_mesh_trgt, process=False)
    mapper = inv_remapper if mode == 'test' else None
    mesh_transfer = project_to_mesh(mesh, mesh_trgt, 'semseg', inv_remapper=mapper)
    semseg = mesh_transfer.vertex_attributes['label']
    # save as txt for benchmark evaluation
    save_txt_path = os.path.join(save_path, 'txt')
    os.makedirs(save_txt_path, exist_ok=True)
    np.savetxt(os.path.join(save_txt_path, '%s.txt'%scene), semseg, fmt='%d')
    
    save_transfer_path = os.path.join(save_path, 'mesh')
    os.makedirs(save_transfer_path, exist_ok=True)
    mesh_transfer.export(os.path.join(save_transfer_path, '%s_transfer.ply'%scene))
    
    pred_ids = read_label(os.path.join(save_transfer_path, '%s_transfer.ply'%scene), is_gt=False)
    gt_ids = read_label(file_mesh_trgt, is_gt=True) if mode != 'test' else None
    

    return scene, (pred_ids, gt_ids)


def main_worker(info_files, mode):
    pred_gt = {}
    for i, info_file in enumerate(info_files):
        scene, temp = process(info_file, i, len(info_files), mode)
        if temp is not None:
            pred_gt[scene] = temp
    return pred_gt


@ray.remote(num_cpus=args.num_workers + 1, num_gpus=(1 / args.n_proc))
def process_with_single_worker(info_files, mode):
    pred_gt = main_worker(info_files, mode)
    return pred_gt


def split_list(_list, n):
    assert len(_list) >= n
    ret = [[] for _ in range(n)]
    for idx, item in enumerate(_list):
        ret[idx % n].append(item)
    return ret


def read_scenes(path, mode='val'):
    if args.dataset == 'scannet':
        with open(os.path.join(path, 'scannetv2_{}.txt'.format(mode))) as f:
            split_files = f.readlines()
        
        split_files = [i.strip() for i in split_files]
    
    return sorted(split_files)


def main():
    all_proc = args.n_proc #  * args.n_gpu

    ray.init(num_cpus=all_proc * (args.num_workers + 1), num_gpus=args.n_gpu)

    info_files = read_scenes(args.data_path, args.mode)
    split = 'scans' if args.mode != 'test' else 'scans_test'
    args.data_path = args.data_path + split
    args.gt_path = args.gt_path + split
    

    if all_proc > 1:
        info_files = split_list(info_files, all_proc)

        ray_worker_ids = []
        for w_idx in range(all_proc):
            ray_worker_ids.append(process_with_single_worker.remote(info_files[w_idx], args.mode))

        results = ray.get(ray_worker_ids)
    else:
        results = main_worker(info_files, args.mode)

    pred_gt = {}
    for r in results:
        pred_gt.update(r)
    
    if args.mode != 'test':
        preds = np.concatenate([i[0] for i in pred_gt.values()])
        gts = np.concatenate([i[1] for i in pred_gt.values()])
        mean_iou = compute_miou(preds, gts, stdout=True, dataset=args.dataset)


if __name__ == "__main__":
    main()