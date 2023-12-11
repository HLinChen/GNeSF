import os
import csv
import logging
import os.path as osp
import numpy as np
import json

from detectron2.data import DatasetCatalog, MetadataCatalog

from .register_hypersim_semantic import label_mapper

logger = logging.getLogger(__name__)


CLASSES = [ # 21
		# 'undefined',
		'wall',
		'floor',
		'cabinet',
		'bed',
		'chair',
		'sofa',
		'table',
		'door',
		'window',
		'bookshelf',
		'picture',
		'counter',
		'desk',
		'curtain',
		'refrigerator',
		'shower curtain',
		'toilet',
		'sink',
		'bathtub',
		'otherfurniture'
		'undefined',
	]


PALETTE = [
            # (0, 0, 0),
            (174, 199, 232), (152, 223, 138), (31, 119, 180), (255, 187, 120), (188, 189, 34),
            (140, 86, 75), (255, 152, 150), (214, 39, 40), (197, 176, 213), (148, 103, 189),
            (196, 156, 148), (23, 190, 207), (247, 182, 210), (219, 219, 141), (255, 127, 14),
            (158, 218, 229), (44, 160, 44), (112, 128, 144), (227, 119, 194), (82, 84, 163),
            (0, 0, 0),
        ]


SCANNET_NAME = 'scannet'
SAMPLE_STEP = 2


def ids_to_nyu40(dir=None):
    """ Returns a dict mapping scannet Ids to NYU40 Ids

    Args:
        path: Path to the original scannet data. 
            This is used to get scannetv2-labels.combined.tsv

    Returns:
        mapping: A dict from ints to ints
            example:
                {1: 1,
                2: 5,
                22: 23}

    """
    if dir is None:
        path = os.path.join(os.path.expanduser(os.getenv("DETECTRON2_DATASETS", "datasets")), SCANNET_NAME)
    else: path = dir
    mapping = {}
    with open(os.path.join(path, 'scannetv2-labels.combined.tsv')) as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter='\t')
        for i, line in enumerate(tsvreader):
            if i==0:
                continue
            scannet_id, nyu40id = int(line[0]), int(line[4])
            mapping[scannet_id] = nyu40id
            
            
    for i in range(40):
        if mapping.get(i, None) == None:
            mapping[i] = 0
    return mapping


def scannet_main_ids():
    # Ignore indices 13, 15, 17, 18, 19, 20, 21, 22, 23, 25, 26. 27. 29. 30. 31. 32, 35. 37. 38, 40
    ids = [
		0,
		1,   
		2,   
		3,   
		4,   
		5,   
		6,   
		7,   
		8,   
		9,   
		10,  
		11,  
		12,  
		14,  
		16,  
		24,  
		28,  
		33,  
		34,  
		36,  
		39  
	]
    # mapping = {key: value for value, key in enumerate(ids)}
    return ids


def ids_to_nyu20(dir=None):

    map40 = ids_to_nyu40(dir)
    main_ids = scannet_main_ids()

    # # original keys to 21 main values in range [0, 40]
    mapping = {k: (v if v in main_ids else 0) for k, v in map40.items()}

    # # original keys to 21 main values in range [0, 20]
    mapping = {k: main_ids.index(v) for k, v in mapping.items()}

    max_id = max(mapping.keys())
    # Map relevant classes to {0,1,...,19}, and ignored classes to 255
    remapper = np.ones(max_id+1) * 0
    for k, v in mapping.items():
        # remapper[k] = v
        remapper[k] = v - 1
    
    remapper[remapper==-1] = 255
    
    return remapper


def scannet_ids_to_cls30(dir=None):
    
    map40 = ids_to_nyu40(dir)
    main_ids = label_mapper.keys()

    # # original keys to 21 main values in range [0, 40]
    # mapping = {k: (v if v in main_ids.keys() else 0) for k, v in map40.items()}
    mapping = {k: (v if v in main_ids else 0) for k, v in map40.items()}

    # # original keys to 30 main values in range [0, 29]
    mapping = {k: label_mapper[v] for k, v in mapping.items()}

    # print(mapping)
    # return mapping

    max_id = max(mapping.keys())
    # Map relevant classes to {0,1,...,19}, and ignored classes to 255
    remapper = np.ones(max_id+1) * (0) # 255
    # for i, x in enumerate(main_ids):
    #     remapper[x] = i
    for k, v in mapping.items():
        remapper[k] = v + 1
    
    return remapper
    

def _load_data(scene_dir, sample_step=10):
    total_ids = os.listdir(osp.join(scene_dir, 'color'))
    total_ids = sorted([int(osp.splitext(frame)[0]) for frame in total_ids])
    sample_ids = total_ids[::sample_step]

    img_infos = []
    for i, frame_id in enumerate(sample_ids):
        img_dir = osp.join(scene_dir, 'color', '{}.jpg'.format(frame_id))
        seg_dir = osp.join(scene_dir, 'label-filt', '{}.png'.format(frame_id))
        img_infos.append(
            {
                "file_name": img_dir,
                "sem_seg_file_name": seg_dir,
            }
        )
    return img_infos


def _load_data_ids(scene_dir, mode='train', ids=None, scene=None):
    sample_ids = ids

    img_infos = []
    for i, frame_id in enumerate(sample_ids):
        img_dir = osp.join(scene_dir, 'color', '{}.jpg'.format(frame_id))
        # nerf:
        # img_dir = osp.join(path, scene, '{}.png'.format(frame_id))
        seg_dir = osp.join(scene_dir, 'label-filt', '{}.png'.format(frame_id))
        img_infos.append(
            {
                "file_name": img_dir,
                "sem_seg_file_name": seg_dir,
            }
        )
    return img_infos


def _get_scene(dir='./data', mode='val'):
    file_path = '{}/scannetv2_{}.txt'.format(dir, mode)
    
    with open(file_path) as f:
        scenes = f.readlines()
    
    scenes = [i.strip() for i in scenes]
    scenes = sorted(scenes)
    
    return scenes


def load_scannet_semantic(data_root, mode='train', sample_step=2):
    img_infos = []
    
    scenes = _get_scene(data_root, mode)
    # for i, scene in enumerate(tqdm.tqdm(scenes)):
    for scene in scenes:
        scene_dir = osp.join(data_root, 'scans', scene)
        data = _load_data(scene_dir, sample_step) # pose c2w
        
        img_infos.extend(data)

    logger.info("Loaded {} images from {}".format(len(img_infos), data_root))
    return img_infos


def load_scannet_semantic_json(data_root, mode='train'):
    img_infos = []
    
    scenes = _get_scene(data_root, mode)
    with open('data/scannet_{}.json'.format(mode), 'r') as f:
        data_all = json.load(f)
    # for i, scene in enumerate(tqdm.tqdm(scenes)):
    for scene in scenes:
        scene_dir = osp.join(data_root, 'scans', scene)
        data = _load_data_ids(scene_dir, mode, data_all[scene], scene=scene)
        # pose c2w
        
        img_infos.extend(data)

    # img_infos = img_infos[::20]
    logger.info("Loaded {} images from {}".format(len(img_infos), data_root))
    return img_infos


_RAW_SCANNET_SPLITS_STEP = {
    "scannet_sem_seg_train_5": ("train"),
    "scannet_sem_seg_val_5": ("val"),
}

def register_scannet_step(root):
    root = os.path.join(root, SCANNET_NAME)
    
    for sem_key, mode in _RAW_SCANNET_SPLITS_STEP.items():
        sample_step = SAMPLE_STEP if mode == 'train' else 10
        DatasetCatalog.register(
            sem_key, lambda x=root, y=mode: load_scannet_semantic(x, y, sample_step)
        )
        MetadataCatalog.get(sem_key).set(
            stuff_classes=CLASSES,
            # evaluator_type="sem_seg",
            evaluator_type="trans_label_sem_seg",
            ignore_label=255,
            stuff_colors=PALETTE,
            # **meta,
        )


_RAW_SCANNET_SPLITS = {
    "scannet_sem_seg_train": ("train"),
    "scannet_sem_seg_val": ("val"),
}

def register_scannet(root):
    root = os.path.join(root, 'scannet')
    for sem_key, mode in _RAW_SCANNET_SPLITS.items():
        DatasetCatalog.register(
            sem_key, lambda x=root, y=mode: load_scannet_semantic_json(x, y)
        )
        MetadataCatalog.get(sem_key).set(
            stuff_classes=CLASSES,
            # evaluator_type="sem_seg",
            evaluator_type="trans_label_sem_seg",
            ignore_label=255,
            stuff_colors=PALETTE,
        )
        

_RAW_SCANNET_SPLITS_FULL = {
    "scannet_sem_seg_train_full": ("train"),
    "scannet_sem_seg_val_full": ("val"),
}


def register_scannet_full(root):
    root = os.path.join(root, 'scannet')
    for sem_key, mode in _RAW_SCANNET_SPLITS_FULL.items():
        sample_step = 1 # if mode == 'train' else 50
        DatasetCatalog.register(
            sem_key, lambda x=root, y=mode: load_scannet_semantic(x, y, sample_step)
        )
        MetadataCatalog.get(sem_key).set(
            stuff_classes=CLASSES,
            evaluator_type="trans_label_sem_seg",
            ignore_label=255,
            stuff_colors=PALETTE,
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_scannet(_root)
register_scannet_step(_root)
register_scannet_full(_root)
