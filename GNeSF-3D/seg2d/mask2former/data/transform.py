from ast import Raise
import numpy as np
from fvcore.transforms.transform import Transform

from .datasets.register_hypersim_semantic import hypersim_ids_to_cls30
# from .datasets.register_coco_stuff_164k import cocoid_to_scannetid
from .datasets.register_scannet_semantic import ids_to_nyu20 # scannet_ids_to_cls30, 


class RemapSemTransform(Transform):
    """
    Resize the image to a target size.
    """

    def __init__(self, dataset='scannet', num_cls=30): # , new_h, new_w
        """
        Args:
            h, w (int): original image size
            new_h, new_w (int): new image size
            interp: PIL interpolation methods, defaults to bilinear.
        """
        # TODO decide on PIL vs opencv
        super().__init__()
        self._set_attributes(locals())
        # self.label_mapping = ids_to_nyu40()
        self.num_cls = num_cls
        # print('dataset', dataset)
        if dataset == 'scannet':
            # self.label_mapping = scannet_ids_to_cls30()
            self.label_mapping = ids_to_nyu20()
            self.ignore_label = 0
        elif dataset == 'hypersim':
            self.label_mapping = hypersim_ids_to_cls30()
            self.ignore_label = 255
        # elif dataset == 'coco':
        #     self.label_mapping = cocoid_to_scannetid()
        #     self.ignore_label = 255
        else:
            self.label_mapping = None

    def apply_image(self, img):
        return img

    def apply_coords(self, coords):
        return coords

    def apply_segmentation(self, segmentation):
        segmentation_map = segmentation.copy()
        # for scan_id, nyu_id in self.label_mapping.items():
        #     segmentation_map[segmentation==scan_id] = nyu_id
        
        if self.label_mapping is not None:
            # segmentation_map[segmentation==self.ignore_label] = self.num_cls
            segmentation_map = self.label_mapping[segmentation_map.astype(np.int32)]
            # print(segmentation_map.min())
            # segmentation_map[segmentation_map==255] = 0
            # print(segmentation.max())
            # print(segmentation_map)
        else:
            # print(segmentation)
            segmentation_map = segmentation_map.astype(np.float32)
        
        # shape = segmentation.shape
        # segmentation = np.array([self.label_map.get(s, 0) for s in segmentation.flatten()]) # .get(s, 0)
        # semantic = semantic.reshape(shape)
        # return segmentation_map.astype(np.uint8)
        return segmentation_map # .astype(np.int8) .astype(np.int16)
        # return segmentation_map - 1 # .astype(np.int8) .astype(np.int16)

    def inverse(self):
        Raise("No Implementation!")