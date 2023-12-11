import numpy as np
from typing import List, Optional, Union


from detectron2.data import DatasetMapper
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils
from detectron2.config import configurable

from ..transform import RemapSemTransform

__all__ = ["TransLabelDatasetMapper"]

class TransLabelDatasetMapper(DatasetMapper):
    
    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        use_instance_mask: bool = False,
        use_keypoint: bool = False,
        instance_mask_format: str = "polygon",
        keypoint_hflip_indices: Optional[np.ndarray] = None,
        precomputed_proposal_topk: Optional[int] = None,
        recompute_boxes: bool = False,
    ):
        super().__init__(
            is_train = is_train,
            augmentations = augmentations,
            image_format = image_format,
            use_instance_mask = use_instance_mask,
            use_keypoint = use_keypoint,
            instance_mask_format = instance_mask_format,
            keypoint_hflip_indices = keypoint_hflip_indices,
            precomputed_proposal_topk = precomputed_proposal_topk,
            recompute_boxes = recompute_boxes,
        )
        
        
    
    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        augs = utils.build_augmentation(cfg, is_train)
        if cfg.INPUT.CROP.ENABLED and is_train:
            augs.insert(0, T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
            recompute_boxes = cfg.MODEL.MASK_ON
        else:
            recompute_boxes = False
        
        # print('!!!')
        augs.append(RemapSemTransform(dataset=cfg.DATASETS.TEST[0].split('_')[0]))

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "use_instance_mask": cfg.MODEL.MASK_ON,
            "instance_mask_format": cfg.INPUT.MASK_FORMAT,
            "use_keypoint": cfg.MODEL.KEYPOINT_ON,
            "recompute_boxes": recompute_boxes,
        }

        if cfg.MODEL.KEYPOINT_ON:
            ret["keypoint_hflip_indices"] = utils.create_keypoint_hflip_indices(cfg.DATASETS.TRAIN)

        if cfg.MODEL.LOAD_PROPOSALS:
            ret["precomputed_proposal_topk"] = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )
        return ret