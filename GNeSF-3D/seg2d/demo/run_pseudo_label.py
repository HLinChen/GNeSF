import os
import glob
import argparse
import multiprocessing as mp

# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import tempfile
import time
import warnings

import cv2
import numpy as np
import tqdm

import torch
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from mask2former import add_maskformer2_config
from demo import setup_cfg, get_parser, setup_logger
from predictor import VisualizationDemo


def get_parser():
    parser = argparse.ArgumentParser(description="maskformer2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="./configs/ade20k/semantic-segmentation/maskformer2_R50_bs16_160k_32cls.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--data_dir",
        default="/ssd/hlchen/scannet/scans/",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--save_dir",
        # default="/data/hlchen/data/scannet_sudo/scans/",
        default="/hdd/hlchen/data/scannet_sudo/scans/",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    parser.add_argument('--i', default=0, type=int,
        help='index of part for parallel processing')
    parser.add_argument('--n', default=1, type=int,
        help='number of parts to devide data into for parallel processing')
    return parser


def run_pseudo_label_scene(demo, scene_dir, save_dir=None, save_sem=True, save_vis=True):
    save_dir = scene_dir if save_dir is None else save_dir
    img_dir = os.path.join(scene_dir, 'color')
    sem_dir = os.path.join(save_dir, 'semantic')
    vis_dir = os.path.join(save_dir, 'vis')
    if save_sem: os.makedirs(sem_dir, exist_ok=True)
    if save_vis: os.makedirs(vis_dir, exist_ok=True)
    
    # img_p = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
    frame_ids = os.listdir(img_dir)
    frame_ids = [int(os.path.splitext(frame)[0]) for frame in frame_ids]
    frame_ids =  sorted(frame_ids)
    frame_ids =  sorted(frame_ids)[::2]
    
    last_id = frame_ids[-1]
    # if os.path.exists(os.path.join(vis_dir, f'{last_id}.png')): return
    
    # for path in tqdm.tqdm(args.input, disable=not args.output):
    # for path in img_p:
    for id in tqdm.tqdm(frame_ids):
        path = os.path.join(img_dir, f'{id}.jpg')
        # use PIL, to be consistent with evaluation
        img = read_image(path, format="BGR")
        # start_time = time.time()
        
        if save_sem:
            sem = predictions["sem_seg"].argmax(dim=0).cpu().numpy().astype("uint8")
            
            sem_rm = os.path.join(sem_dir, os.path.basename(path))
            if os.path.exists(sem_rm):
                os.system('rm {}'.format(sem_rm))
            save_sem_p = os.path.join(sem_dir, f'{id}.png')
            cv2.imwrite(save_sem_p, sem)
        
        if save_vis:
            predictions, visualized_output = demo.run_on_image(img)
        
            vis_rm = os.path.join(vis_dir, os.path.basename(path))
            if os.path.exists(vis_rm):
                os.system('rm {}'.format(vis_rm))
            save_vis_p = os.path.join(vis_dir, f'{id}.png')
            visualized_output.save(save_vis_p)
        
    return


# src_dir = '/ssd/hlchen/scannet_5/scans/'
# save_dir = '/ssd/hlchen/scannet/scans/'
# cfg_dir = '../configs/ade20k/semantic-segmentation/maskformer2_R50_bs16_160k_32cls.yaml'
# cfg_dir = './configs/ade20k/semantic-segmentation/maskformer2_R50_bs16_160k_32cls.yaml'
# model_dir = '/data/hlchen/output/mask2former/ade_32cls_256head_conv_50queries_mask32_fw1024/model_final.pth'

def main(args, i=0, n=1):
    # torch.cuda.set_device(i)
    # torch.cuda.set_device(0)
    num_gpu = torch.cuda.device_count()
    torch.cuda.set_device(i%num_gpu)
    
    mp.set_start_method("spawn", force=True)
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)
    
    save_sem = False
    save_vis = True
    
    scenes = sorted(os.listdir(args.data_dir))
    scenes = scenes[i::n]
    for scene in tqdm.tqdm(scenes):
        scene_dir = os.path.join(args.data_dir, scene)
        save_dir = os.path.join(args.save_dir, scene)
        run_pseudo_label_scene(demo, scene_dir, save_dir, save_sem=save_sem, save_vis=save_vis)
    
    return


if __name__ == '__main__':
    args = get_parser().parse_args()
    i=args.i
    n=args.n
    main(args, i=i, n=n)
    