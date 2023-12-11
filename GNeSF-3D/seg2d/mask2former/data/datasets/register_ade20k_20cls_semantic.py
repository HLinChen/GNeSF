# Copyright (c) Facebook, Inc. and its affiliates.
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

ADE20K_21_CATEGORIES = [
    {"color": [174, 199, 232], "id": 0, "isthing": 0, "name": "wall"},
    {"color": [152, 223, 138], "id": 1, "isthing": 0, "name": "floor"},
    {"color": [31, 119, 180], "id": 2, "isthing": 1, "name": "cabinet"},
    {"color": [255, 187, 120], "id": 3, "isthing": 1, "name": "bed"},
    {
        "color": [188, 189, 34],
        "id": 4,
        "isthing": 1,
        "name": "chair, swivel chair, armchair",
    },
    {"color": [140, 86, 75], "id": 5, "isthing": 1, "name": "sofa"},
    {
        "color": [247, 182, 210],
        "id": 6,
        "isthing": 1,
        "name": "table, coffee table, desk", # pool table, billiard table, snooker table, 
    },
    {
        "color": [214, 39, 40], 
        "id": 7, 
        "isthing": 1, 
        "name": "door, screen door"
    },
    {"color": [197, 176, 213], "id": 8, "isthing": 1, "name": "window"},
    {
        "color": [23, 190, 207], 
        "id": 9, 
        "isthing": 1, 
        "name": "counter, countertop, buffet, sideboard"
    },
    {"color": [66, 188, 102], "id": 10, "isthing": 1, "name": "bookcase"}, # shelf, 
    {"color": [219, 219, 141], "id": 11, "isthing": 1, "name": "curtain"},
    {"color": [78, 71, 183], "id": 12, "isthing": 0, "name": "ceiling"},
    {"color": [255, 127, 14], "id": 13, "isthing": 1, "name": "refrigerator, icebox"},
    {"color": [91, 163, 138], "id": 14, "isthing": 1, "name": "tv"},
    {
        "color": [44, 160, 44],
        "id": 15,
        "isthing": 1,
        "name": "toilet, can, commode, crapper, potty",
    },
    {"color": [112, 128, 144], "id": 16, "isthing": 1, "name": "sink"},
    {"color": [96, 207, 209], "id": 17, "isthing": 1, "name": "lamp"},
    {"color": [213, 92, 176], "id": 18, "isthing": 1, "name": "bag"},
    {"color": [196, 156, 148], "id": 19, "isthing": 1, "name": "painting, picture"},
    {"color": [227, 119, 194], "id": 20, "isthing": 1, "name": "tub"},
    # {"color": [150, 5, 61], "id": 15, "isthing": 1, "name": "person"},
]

ADE20K_30_CATEGORIES = [
    {"color": [120, 120, 120], "id": 0, "isthing": 0, "name": "wall"},
    {"color": [80, 50, 50], "id": 1, "isthing": 0, "name": "floor"},
    {"color": [224, 5, 255], "id": 2, "isthing": 1, "name": "cabinet"},
    {"color": [204, 5, 255], "id": 3, "isthing": 1, "name": "bed"},
    {
        "color": [204, 70, 3],
        "id": 4,
        "isthing": 1,
        "name": "chair, swivel chair, armchair",
    },
    {"color": [11, 102, 255], "id": 5, "isthing": 1, "name": "sofa"},
    {
        "color": [255, 71, 0],
        "id": 6,
        "isthing": 1,
        "name": "table, pool table, billiard table, snooker table, coffee table",
    },
    {
        "color": [8, 255, 51], 
        "id": 7, 
        "isthing": 1, 
        "name": "door, screen door"
    },
    {"color": [230, 230, 230], "id": 8, "isthing": 1, "name": "window "},
    {
        "color": [235, 12, 255], 
        "id": 9, 
        "isthing": 1, 
        "name": "counter, countertop, buffet, sideboard"
    },
    {"color": [255, 7, 71], "id": 10, "isthing": 1, "name": "bookcase"}, # shelf, 
    {"color": [255, 51, 7], "id": 11, "isthing": 1, "name": "curtain"},
    {"color": [120, 120, 80], "id": 12, "isthing": 0, "name": "ceiling"},
    {"color": [20, 255, 0], "id": 13, "isthing": 1, "name": "refrigerator, icebox"},
    {"color": [0, 255, 194], "id": 14, "isthing": 1, "name": "tv"},
    {"color": [150, 5, 61], "id": 15, "isthing": 1, "name": "person"},
    {
        "color": [0, 255, 132],
        "id": 16,
        "isthing": 1,
        # "name": "toilet, can, commode, crapper, pot, potty, stool, throne",
        "name": "toilet, can, commode, crapper, potty",
    },
    {"color": [0, 163, 255], "id": 17, "isthing": 1, "name": "sink"},
    {"color": [224, 255, 8], "id": 18, "isthing": 1, "name": "lamp"},
    {"color": [70, 184, 160], "id": 19, "isthing": 1, "name": "bag"},
    {"color": [0, 255, 10], "id": 20, "isthing": 1, "name": "bottle"},
    {"color": [0, 255, 10], "id": 21, "isthing": 1, "name": "cup"},         # ??? not in ADE20K
    {"color": [0, 255, 10], "id": 22, "isthing": 1, "name": "keyboard"},    # ??? not in ADE20K
    {"color": [0, 255, 10], "id": 23, "isthing": 1, "name": "mouse"},       # ??? not in ADE20K
    {"color": [255, 163, 0], "id": 24, "isthing": 1, "name": "book"},
    {"color": [255, 163, 0], "id": 25, "isthing": 1, "name": "laptop"},     # ??? not in ADE20K
    # {"color": [0, 255, 173], "id": 74, "isthing": 1, "name": "computer"},
    {"color": [20, 0, 255], "id": 26, "isthing": 0, "name": "blanket, cover"},
    {"color": [0, 235, 255], "id": 27, "isthing": 1, "name": "pillow"},
    {"color": [102, 255, 0], "id": 28, "isthing": 1, "name": "clock"},
    {"color": [102, 255, 0], "id": 29, "isthing": 1, "name": "cellphone"}, # ??? not in ADE20K
]


ADE20K_32_CATEGORIES = [
    {"color": [174, 199, 232], "id": 0, "isthing": 0, "name": "wall"},
    {"color": [152, 223, 138], "id": 1, "isthing": 0, "name": "floor"},
    {"color": [31, 119, 180], "id": 2, "isthing": 1, "name": "cabinet"},
    {"color": [255, 187, 120], "id": 3, "isthing": 1, "name": "bed"},
    {
        "color": [188, 189, 34],
        "id": 4,
        "isthing": 1,
        "name": "chair, swivel chair, armchair",
    },
    {"color": [140, 86, 75], "id": 5, "isthing": 1, "name": "sofa"},
    {
        "color": [255, 152, 150],
        "id": 6,
        "isthing": 1,
        "name": "table, coffee table, desk", # , pool table, billiard table, snooker table , desk
    },
    {
        "color": [214, 39, 40], 
        "id": 7, 
        "isthing": 1, 
        "name": "door, screen door"
    },
    {"color": [197, 176, 213], "id": 8, "isthing": 1, "name": "window"},
    {"color": [148, 103, 189], "id": 9, "isthing": 1, "name": "bookcase"},
    {"color": [196, 156, 148], "id": 10, "isthing": 1, "name": "painting, picture"},
    {
        "color": [23, 190, 207], 
        "id": 11, 
        "isthing": 1, 
        "name": "counter, countertop, buffet, sideboard"
    },
    {"color": [178, 76, 76], "id": 12, "isthing": 0, "name": "blind"}, # , screen
    # {"color": [247, 182, 210], "id": 13, "isthing": 1, "name": "desk"},
    {"color": [66, 188, 102], "id": 13, "isthing": 1, "name": "shelf"},
    {"color": [219, 219, 141], "id": 14, "isthing": 1, "name": "curtain"},
    {
        "color": [140, 57, 197],
        "id": 15,
        "isthing": 1,
        "name": "chest of drawers, chest, bureau, dresser",
    },
    {"color": [202, 185, 52], "id": 16, "isthing": 1, "name": "pillow"},
    {"color": [51, 176, 203], "id": 17, "isthing": 1, "name": "mirror"},
    {"color": [200, 54, 131], "id": 18, "isthing": 0, "name": "blanket, cover"}, # floormat ?
    {"color": [92, 193, 61], "id": 19, "isthing": 1, "name": "clothes"},
    {"color": [78, 71, 183], "id": 20, "isthing": 0, "name": "ceiling"},
    {"color": [172, 114, 82], "id": 21, "isthing": 1, "name": "book"},
    {"color": [255, 127, 14], "id": 22, "isthing": 1, "name": "refrigerator, icebox"},
    {"color": [91, 163, 138], "id": 23, "isthing": 1, "name": "tv"},
    {"color": [140, 153, 101], "id": 24, "isthing": 1, "name": "towel"},
    {"color": [100, 125, 154], "id": 25, "isthing": 1, "name": "box"},
    {"color": [120, 185, 128], "id": 26, "isthing": 1, "name": "person"},
    {
        "color": [44, 160, 44],
        "id": 27,
        "isthing": 1,
        # "name": "toilet, can, commode, crapper, pot, potty, stool, throne",
        "name": "toilet, can, commode, crapper, potty",
    },
    {"color": [112, 128, 144], "id": 28, "isthing": 1, "name": "sink"},
    {"color": [96, 207, 209], "id": 29, "isthing": 1, "name": "lamp"},
    {"color": [227, 119, 194], "id": 30, "isthing": 1, "name": "tub"},
    {"color": [213, 92, 176], "id": 31, "isthing": 1, "name": "bag"},
]


ADE20K_40_CATEGORIES = [
    {"color": [174, 199, 232], "id": 0, "isthing": 0, "name": "wall"},
    {"color": [152, 223, 138], "id": 1, "isthing": 0, "name": "floor"},
    {"color": [31, 119, 180], "id": 2, "isthing": 1, "name": "cabinet"},
    {"color": [255, 187, 120], "id": 3, "isthing": 1, "name": "bed"},
    {
        "color": [188, 189, 34],
        "id": 4,
        "isthing": 1,
        "name": "chair, swivel chair, armchair",
    },
    {"color": [140, 86, 75], "id": 5, "isthing": 1, "name": "sofa"},
    {
        "color": [255, 152, 150],
        "id": 6,
        "isthing": 1,
        "name": "table, coffee table", # , pool table, billiard table, snooker table , desk
    },
    {
        "color": [214, 39, 40], 
        "id": 7, 
        "isthing": 1, 
        "name": "door, screen door"
    },
    {"color": [197, 176, 213], "id": 8, "isthing": 1, "name": "window"},
    {"color": [148, 103, 189], "id": 9, "isthing": 1, "name": "bookcase"},
    {"color": [196, 156, 148], "id": 10, "isthing": 1, "name": "painting, picture"},
    {
        "color": [23, 190, 207], 
        "id": 11, 
        "isthing": 1, 
        "name": "counter, countertop, buffet, sideboard"
    },
    {"color": [178, 76, 76], "id": 12, "isthing": 0, "name": "blind"}, # , screen
    {"color": [247, 182, 210], "id": 13, "isthing": 1, "name": "desk"},
    {"color": [66, 188, 102], "id": 14, "isthing": 1, "name": "shelf"},
    {"color": [219, 219, 141], "id": 15, "isthing": 1, "name": "curtain"},
    {
        "color": [140, 57, 197],
        "id": 16,
        "isthing": 1,
        "name": "chest of drawers, chest, bureau, dresser",
    },
    {"color": [202, 185, 52], "id": 17, "isthing": 1, "name": "pillow"},
    {"color": [51, 176, 203], "id": 18, "isthing": 1, "name": "mirror"},
    {"color": [200, 54, 131], "id": 19, "isthing": 0, "name": "blanket, cover"}, # floormat ?
    {"color": [92, 193, 61], "id": 20, "isthing": 1, "name": "clothes"},
    {"color": [78, 71, 183], "id": 21, "isthing": 0, "name": "ceiling"},
    {"color": [172, 114, 82], "id": 22, "isthing": 1, "name": "book"},
    {"color": [255, 127, 14], "id": 23, "isthing": 1, "name": "refrigerator, icebox"},
    {"color": [91, 163, 138], "id": 24, "isthing": 1, "name": "tv"},
    # {"color": [153, 98, 156], "id": 25, "isthing": 1, "name": "paper"}, not in ade20k
    {"color": [140, 153, 101], "id": 26, "isthing": 1, "name": "towel"},
    # {"color": [158, 218, 229], "id": 27, "isthing": 1, "name": "shower curtain"}, not in ade20k
    {"color": [100, 125, 154], "id": 28, "isthing": 1, "name": "box"},
    # {"color": [178, 127, 135], "id": 29, "isthing": 1, "name": "whiteboard"}, not in ade20k
    {"color": [120, 185, 128], "id": 30, "isthing": 1, "name": "person"},
    # {"color": [146, 111, 194], "id": 31, "isthing": 1, "name": "night stand"}, not in ade20k
    {
        "color": [44, 160, 44],
        "id": 32,
        "isthing": 1,
        # "name": "toilet, can, commode, crapper, pot, potty, stool, throne",
        "name": "toilet, can, commode, crapper, potty",
    },
    {"color": [112, 128, 144], "id": 33, "isthing": 1, "name": "sink"},
    {"color": [96, 207, 209], "id": 34, "isthing": 1, "name": "lamp"},
    {"color": [227, 119, 194], "id": 35, "isthing": 1, "name": "tub"},
    {"color": [213, 92, 176], "id": 36, "isthing": 1, "name": "bag"},
    # {"color": [94, 106, 211], "id": 37, "isthing": 1, "name": "otherstructure"}, not in ade20k
    # {"color": [82, 84, 163], "id": 38, "isthing": 1, "name": "otherfurniture"}, not in ade20k
    # {"color": [100, 85, 144], "id": 39, "isthing": 1, "name": "otherprop"}, not in ade20k
]


ADE20k_21CLS_COLORS = [k["color"] for k in ADE20K_21_CATEGORIES]
ADE20k_30CLS_COLORS = [k["color"] for k in ADE20K_30_CATEGORIES]
ADE20k_32CLS_COLORS = [k["color"] for k in ADE20K_32_CATEGORIES]
ADE20k_40CLS_COLORS = [k["color"] for k in ADE20K_40_CATEGORIES]


def _get_ade20k_meta(CATEGORIES):
    # Id 0 is reserved for ignore_label, we change ignore_label for 0
    # to 255 in our pre-processing, so all ids are shifted by 1.
    stuff_ids = [k["id"] for k in CATEGORIES]
    # assert len(stuff_ids) == 21, len(stuff_ids)

    # For semantic segmentation, this mapping maps from contiguous stuff id
    # (in [0, 91], used in models) to ids in the dataset (used for processing results)
    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}
    stuff_classes = [k["name"] for k in CATEGORIES]

    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
    }
    return ret


def register_ade20k_21cls(root):
    root = os.path.join(root, "ADEChallengeData2016")
    meta = _get_ade20k_meta(ADE20K_21_CATEGORIES)
    for name, dirname in [("train", "training"), ("val", "validation")]:
        image_dir = os.path.join(root, "images", dirname)
        gt_dir = os.path.join(root, "annotations_detectron2_21", dirname)
        # gt_dir = os.path.join(root, "annotations_detectron2", dirname)
        name = f"ade20k_21cls_sem_seg_{name}"
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="jpg")
        )
        MetadataCatalog.get(name).set(
            stuff_classes=meta["stuff_classes"][:],
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
            stuff_colors=ADE20k_21CLS_COLORS[:],
        )


def register_ade20k_32cls(root):
    root = os.path.join(root, "ADEChallengeData2016")
    meta = _get_ade20k_meta(ADE20K_32_CATEGORIES)
    for name, dirname in [("train", "training"), ("val", "validation")]:
        image_dir = os.path.join(root, "images", dirname)
        gt_dir = os.path.join(root, "annotations_detectron2_32", dirname)
        # gt_dir = os.path.join(root, "annotations_detectron2", dirname)
        name = f"ade20k_32cls_sem_seg_{name}"
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="jpg")
        )
        MetadataCatalog.get(name).set(
            stuff_classes=meta["stuff_classes"][:],
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
            stuff_colors=ADE20k_32CLS_COLORS[:],
        )



def register_ade20k_40cls(root):
    root = os.path.join(root, "ADEChallengeData2016")
    meta = _get_ade20k_meta(ADE20K_40_CATEGORIES)
    for name, dirname in [("train", "training"), ("val", "validation")]:
        image_dir = os.path.join(root, "images", dirname)
        gt_dir = os.path.join(root, "annotations_detectron2_40", dirname)
        # gt_dir = os.path.join(root, "annotations_detectron2", dirname)
        name = f"ade20k_40cls_sem_seg_{name}"
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="jpg")
        )
        MetadataCatalog.get(name).set(
            stuff_classes=meta["stuff_classes"][:],
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
            stuff_colors=ADE20k_32CLS_COLORS[:],
        )

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_ade20k_21cls(_root)
register_ade20k_32cls(_root)
register_ade20k_40cls(_root)


# MetadataCatalog.get("ade20k_21cls_sem_seg_train").set(
#     stuff_colors=ADE20k_21CLS_COLORS[:],
# )

# MetadataCatalog.get("ade20k_21cls_sem_seg_val").set(
#     stuff_colors=ADE20k_21CLS_COLORS[:],
# )