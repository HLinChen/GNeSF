# Copyright (c) Facebook, Inc. and its affiliates.
import os
import logging
import numpy as np

from detectron2.data import DatasetCatalog, MetadataCatalog


logger = logging.getLogger(__name__)


HYPERSIM_CATEGORIES = [
    {"color": [174, 199, 232], "id": 0, "isthing": 0, "name": "wall"},
    {"color": [152, 223, 138], "id": 1, "isthing": 0, "name": "floor"},
    {"color": [31, 119, 180], "id": 2, "isthing": 1, "name": "cabinet"},
    {"color": [255, 187, 120], "id": 3, "isthing": 1, "name": "bed"},
    {
        "color": [188, 189, 34],
        "id": 4,
        "isthing": 1,
        "name": "chair",
    },
    {"color": [140, 86, 75], "id": 5, "isthing": 1, "name": "sofa"},
    {
        "color": [255, 152, 150],
        "id": 6,
        "isthing": 1,
        "name": "table", # , pool table, billiard table, snooker table , desk
    },
    {
        "color": [214, 39, 40], 
        "id": 7, 
        "isthing": 1, 
        "name": "door"
    },
    {"color": [197, 176, 213], "id": 8, "isthing": 1, "name": "window"},
    {"color": [148, 103, 189], "id": 9, "isthing": 1, "name": "bookshelf"},
    {"color": [196, 156, 148], "id": 10, "isthing": 1, "name": "picture"},
    {
        "color": [23, 190, 207], 
        "id": 11, 
        "isthing": 1, 
        "name": "counter"
    },
    {"color": [178, 76, 76], "id": 12, "isthing": 0, "name": "blinds"}, # , screen
    {"color": [247, 182, 210], "id": 13, "isthing": 1, "name": "desk"},
    {"color": [66, 188, 102], "id": 14, "isthing": 1, "name": "shelves"},
    {"color": [219, 219, 141], "id": 15, "isthing": 1, "name": "curtain"},
    {
        "color": [140, 57, 197],
        "id": 16,
        "isthing": 1,
        "name": "dresser",
    },
    {"color": [202, 185, 52], "id": 17, "isthing": 1, "name": "pillow"},
    {"color": [51, 176, 203], "id": 18, "isthing": 1, "name": "mirror"},
    {"color": [200, 54, 131], "id": 19, "isthing": 0, "name": "floor mat"}, # floormat ?
    {"color": [92, 193, 61], "id": 20, "isthing": 1, "name": "clothes"},
    {"color": [78, 71, 183], "id": 21, "isthing": 0, "name": "ceiling"},
    {"color": [172, 114, 82], "id": 22, "isthing": 1, "name": "books"},
    {"color": [255, 127, 14], "id": 23, "isthing": 1, "name": "refridgerator"},
    {"color": [91, 163, 138], "id": 24, "isthing": 1, "name": "television"},
    {"color": [153, 98, 156], "id": 25, "isthing": 1, "name": "paper"},
    {"color": [140, 153, 101], "id": 26, "isthing": 1, "name": "towel"},
    {"color": [158, 218, 229], "id": 27, "isthing": 1, "name": "shower curtain"},
    {"color": [100, 125, 154], "id": 28, "isthing": 1, "name": "box"},
    {"color": [178, 127, 135], "id": 29, "isthing": 1, "name": "whiteboard"},
    {"color": [120, 185, 128], "id": 30, "isthing": 1, "name": "person"},
    {"color": [146, 111, 194], "id": 31, "isthing": 1, "name": "night stand"},
    {
        "color": [44, 160, 44],
        "id": 32,
        "isthing": 1,
        "name": "toilet",
    },
    {"color": [112, 128, 144], "id": 33, "isthing": 1, "name": "sink"},
    {"color": [96, 207, 209], "id": 34, "isthing": 1, "name": "lamp"},
    {"color": [227, 119, 194], "id": 35, "isthing": 1, "name": "bathtub"},
    {"color": [213, 92, 176], "id": 36, "isthing": 1, "name": "bag"},
    {"color": [94, 106, 211], "id": 37, "isthing": 1, "name": "otherstructure"},
    {"color": [82, 84, 163], "id": 38, "isthing": 1, "name": "otherfurniture"},
    {"color": [100, 85, 144], "id": 39, "isthing": 1, "name": "otherprop"},
]


HYPERSIM_CATEGORIES_28 = [
    {"color": [174, 199, 232], "id": 0, "isthing": 0, "name": "wall"},
    {"color": [152, 223, 138], "id": 1, "isthing": 0, "name": "floor"},
    {"color": [31, 119, 180], "id": 2, "isthing": 1, "name": "cabinet"},
    {"color": [255, 187, 120], "id": 3, "isthing": 1, "name": "bed"},
    {
        "color": [188, 189, 34],
        "id": 4,
        "isthing": 1,
        "name": "chair",
    },
    {"color": [140, 86, 75], "id": 5, "isthing": 1, "name": "sofa"},
    {
        "color": [255, 152, 150],
        "id": 6,
        "isthing": 1,
        "name": "desk, table", # , pool table, billiard table, snooker table , desk
    },
    {
        "color": [214, 39, 40], 
        "id": 7, 
        "isthing": 1, 
        "name": "door"
    },
    {"color": [197, 176, 213], "id": 8, "isthing": 1, "name": "blinds, window"},
    {"color": [148, 103, 189], "id": 9, "isthing": 1, "name": "bookshelf, shleves"},
    {"color": [196, 156, 148], "id": 10, "isthing": 1, "name": "picture"},
    {
        "color": [23, 190, 207], 
        "id": 11, 
        "isthing": 1, 
        "name": "counter"
    },
    {"color": [158, 218, 229], "id": 12, "isthing": 0, "name": "curtain, shower curtain"}, # blinds ?
    {"color": [255, 127, 14], "id": 13, "isthing": 1, "name": "refridgerator"},
    {
        "color": [44, 160, 44],
        "id": 14,
        "isthing": 1,
        "name": "toilet",
    },
    {"color": [112, 128, 144], "id": 15, "isthing": 1, "name": "sink"},
    {"color": [227, 119, 194], "id": 16, "isthing": 1, "name": "bathtub"},
    
    {"color": [78, 71, 183], "id": 17, "isthing": 0, "name": "ceiling"},
    {"color": [91, 163, 138], "id": 18, "isthing": 1, "name": "television"},
    {"color": [202, 185, 52], "id": 19, "isthing": 1, "name": "pillow"},
    {"color": [51, 176, 203], "id": 20, "isthing": 1, "name": "mirror"},
    {"color": [200, 54, 131], "id": 21, "isthing": 0, "name": "floor mat"}, # floormat ?
    {"color": [96, 207, 209], "id": 22, "isthing": 1, "name": "lamp"},
    
    {"color": [172, 114, 82], "id": 23, "isthing": 1, "name": "books"},
    {"color": [153, 98, 156], "id": 24, "isthing": 1, "name": "paper"},
    {"color": [140, 153, 101], "id": 25, "isthing": 1, "name": "towel"},
    {"color": [100, 125, 154], "id": 26, "isthing": 1, "name": "box"},
    {"color": [213, 92, 176], "id": 27, "isthing": 1, "name": "bag"},
    # {"color": [178, 127, 135], "id": 28, "isthing": 1, "name": "whiteboard"},   # nan
    # {"color": [146, 111, 194], "id": 29, "isthing": 1, "name": "night stand"}, # nan
    {"color": [0, 0, 0], "id": 28, "isthing": 1, "name": "void"},
]


label_mapper = {
    0: 0, # wall
    1: 1, # floor
    2: 2, # cabinet
    3: 3, # bed
    4: 4, # chair
    5: 5, # sofa
    6: 6, # table, desk
    7: 7, # door
    8: 8, # window
    9: 9, # bookshelf, shleves
    10: 10, # picture
    11: 11, # counter
    12: 8, # blinds, window
    13: 6, # desk, table
    14: 9, # shleves, bookshelf
    15: 12, # curtain
    # 16: 255, # dresser
    23: 13, # refridgerator
    27: 12, # shower curtain -> curtain
    32: 14, # toilet
    33: 15, # sink
    35: 16, # bathtub
    21: 17, # celling
    24: 18, # television
    17: 19, # pillow
    18: 20, # mirror
    19: 21, # floor mat
    34: 22, # lamp
    
    22: 23, # books
    25: 24, # paper
    26: 25, # towel
    28: 26, # box
    36: 27, # bag
    # 29: 28, # whiteboard
    # 31: 29, # night stand
}


HYPERSIM_COLORS = [k["color"] for k in HYPERSIM_CATEGORIES]


HYPERSIM_COLORS_28 = [k["color"] for k in HYPERSIM_CATEGORIES_28]


def _get_hypersim_meta(CATEGORIES):
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


def hypersim_ids_to_cls30():
    main_ids = label_mapper.keys()

    # # original keys to 21 main values in range [0, 40]
    # mapping = {k: (v if v in main_ids.keys() else 0) for k, v in map40.items()}
    # mapping = {k: (-1 if k in main_ids else 30) for k in range(40)}

    # # # original keys to 30 main values in range [0, 29]
    # mapping = {k: label_mapper[v] for k, v in mapping.items()}
    mapping = label_mapper

    # print(mapping)
    # return mapping

    max_id = 39 # max(mapping.keys())
    # Map relevant classes to {0,1,...,19}, and ignored classes to 255
    remapper = np.ones(max_id+1) * (28) # 255
    # for i, x in enumerate(main_ids):
    #     remapper[x] = i
    for k, v in mapping.items():
        remapper[k] = v
    
    return remapper


def _get_scene(dir='./data', mode='val'):
    file_path = '{}/{}.txt'.format(dir, mode)
    # file_path = 'data/debug.txt'
    # file_path = 'data/{}.txt'.format(mode)
    
    with open(file_path) as f:
        scenes = f.readlines()
    
    scenes = [i.strip() for i in scenes]
    scenes = sorted(scenes)
    
    return scenes


def _load_data(scene_dir):
    total_ids = os.listdir(os.path.join(scene_dir, 'color'))
    total_ids = sorted([int(os.path.splitext(frame)[0]) for frame in total_ids])
    sample_ids = total_ids

    img_infos = []
    for i, frame_id in enumerate(sample_ids):
        img_dir = os.path.join(scene_dir, 'color', '{}.jpg'.format(frame_id))
        seg_dir = os.path.join(scene_dir, 'label-filt', '{}.png'.format(frame_id))
        img_infos.append(
            {
                "file_name": img_dir,
                "sem_seg_file_name": seg_dir,
            }
        )
    # img_infos = sorted(img_infos, key=lambda x: x['filename'])
    return img_infos


def load_hypersim_sem_seg(data_root, mode='train'):
    img_infos = []
    
    scenes = _get_scene(data_root, mode)
    # for i, scene in enumerate(tqdm.tqdm(scenes)):
    for scene in scenes:
        scene_dir = os.path.join(data_root, 'scenes', scene)
        data = _load_data(scene_dir) # pose c2w
        
        img_infos.extend(data)

    logger.info("Loaded {} images from {}".format(len(img_infos), data_root))
    return img_infos


def register_hypersim(root):
    root = os.path.join(root, "hypersim_sim")
    meta = _get_hypersim_meta(HYPERSIM_CATEGORIES)
    for mode, dirname in [("train", "training"), ("val", "validation"), ("test", "testing")]:
        # scenes = _get_scene(root, name)
        # image_dir = os.path.join(root, "images", dirname)
        # gt_dir = os.path.join(root, "annotations_detectron2_32", dirname)
        # gt_dir = os.path.join(root, "annotations_detectron2", dirname)
        name = f"hypersim_sem_seg_{mode}"
        DatasetCatalog.register(
            name, lambda x=root, y=mode: load_hypersim_sem_seg(x, y)
        )
        MetadataCatalog.get(name).set(
            stuff_classes=meta["stuff_classes"][:],
            # image_root=image_dir,
            # sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            # ignore_label=255,
            stuff_colors=HYPERSIM_COLORS[:],
        )


def register_hypersim_30(root):
    root = os.path.join(root, "hypersim_sim")
    meta = _get_hypersim_meta(HYPERSIM_CATEGORIES_28)
    for mode, dirname in [("train", "training"), ("val", "validation"), ("test", "testing")]:
        # scenes = _get_scene(root, name)
        # image_dir = os.path.join(root, "images", dirname)
        # gt_dir = os.path.join(root, "annotations_detectron2_32", dirname)
        # gt_dir = os.path.join(root, "annotations_detectron2", dirname)
        name = f"hypersim_28_sem_seg_{mode}"
        DatasetCatalog.register(
            name, lambda x=root, y=mode: load_hypersim_sem_seg(x, y)
        )
        MetadataCatalog.get(name).set(
            stuff_classes=meta["stuff_classes"][:],
            # image_root=image_dir,
            # sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
            stuff_colors=HYPERSIM_COLORS_28[:],
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_hypersim(_root)

register_hypersim_30(_root)


# MetadataCatalog.get("hypersim_21cls_sem_seg_train").set(
#     stuff_colors=HYPERSIM_21CLS_COLORS[:],
# )

# MetadataCatalog.get("hypersim_21cls_sem_seg_val").set(
#     stuff_colors=HYPERSIM_21CLS_COLORS[:],
# )