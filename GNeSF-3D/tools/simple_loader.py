import torch
import numpy as np
import os
import cv2

from tools.semantic_mapper import ids_to_scannet21
from seg2d.mask2former.data.datasets.register_scannet_semantic import ids_to_nyu20


def collate_fn(list_data):
    cam_pose, depth_im, _ = list_data
    # Concatenate all lists
    return cam_pose, depth_im, _


def collate_fn(list_data):
    cam_pose, depth_im, _, sem_im = list_data
    # Concatenate all lists
    return cam_pose, depth_im, _, sem_im


class ScanNetDataset(torch.utils.data.Dataset):
    """Pytorch Dataset for a single scene. getitem loads individual frames"""

    def __init__(self, n_imgs, scene, data_path, max_depth, id_list=None, enable_sem=False, num_cls=20):
        """
        Args:
        """
        self.n_imgs = n_imgs
        self.scene = scene
        self.data_path = data_path
        self.max_depth = max_depth
        self.enable_sem = enable_sem
        if id_list is None:
            self.id_list = [i for i in range(n_imgs)]
        else:
            self.id_list = id_list
        self.label_mapping = ids_to_nyu20() if num_cls == 20 else ids_to_scannet21()
        self.label_mapping[self.label_mapping==255] = -1

    def __len__(self):
        return self.n_imgs

    def __getitem__(self, id):
        """
        Returns:
            dict of meta data and images for a single frame
        """
        id = self.id_list[id]
        cam_pose = np.loadtxt(os.path.join(self.data_path, self.scene, "pose", str(id) + ".txt"), delimiter=' ')

        # Read depth image and camera pose
        depth_im = cv2.imread(os.path.join(self.data_path, self.scene, "depth", str(id) + ".png"), -1).astype(
            np.float32)
        depth_im /= 1000.  # depth is saved in 16-bit PNG in millimeters
        depth_im[depth_im > self.max_depth] = 0

        # Read RGB image
        color_image = cv2.cvtColor(cv2.imread(os.path.join(self.data_path, self.scene, "color", str(id) + ".jpg")),
                                   cv2.COLOR_BGR2RGB)
        color_image = cv2.resize(color_image, (depth_im.shape[1], depth_im.shape[0]), interpolation=cv2.INTER_AREA)
        
        if self.enable_sem:
            sem_image = cv2.imread(os.path.join(self.data_path, self.scene, "label-filt", str(id) + ".png"), -1).astype(np.int32)
            sem_image = cv2.resize(sem_image, (depth_im.shape[1], depth_im.shape[0]), interpolation=cv2.INTER_NEAREST)
            sem_image = self.label_mapping[sem_image].astype(np.int32)
        else:
            sem_image = None

        return cam_pose, depth_im, color_image, sem_image

