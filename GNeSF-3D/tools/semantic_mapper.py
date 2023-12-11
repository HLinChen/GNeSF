import os
import csv

import numpy as np
from pathlib import Path


CLASSES_21 = [ # 21
		'wall',
		'floor',
		'cabinet',
		'bed',
		'chair',
		'sofa',
		'table',
		'door',
		'window',
		'counter',
		'shelves',
		'curtain',
		'ceiling',
		'refrigerator',
		'television',
		'person',
		'toilet',
		'sink',
		'lamp',
		'bag',
		'otherprop',
		# 'undefined',
	]


PALETTE_21 = [
            (174, 199, 232), (152, 223, 138), (31, 119, 180), (255, 187, 120), (188, 189, 34),
            (140, 86, 75), (255, 152, 150), (214, 39, 40), (197, 176, 213), (148, 103, 189),
            (196, 156, 148), (23, 190, 207), (247, 182, 210), (219, 219, 141), (255, 127, 14),
            (158, 218, 229), (44, 160, 44), (112, 128, 144), (227, 119, 194), (82, 84, 163), (82, 84, 163),
            (0, 0, 0),
        ]


# fix ambiguous / incorrect annotations
scene_specific_fixes_objectid = {
    "room_0": {},
    "room_1": {},
    "room_2": {},
    "office_0": {
        3: 15,
        30: 15,
    },
    "office_1": {
    },
    "office_2": {
        69: 15,
        2: 15,
        0: 8,
        3: 8
    },
    "office_3": {
        55: 6,
        8: 6,
        12: 21,
        88: 15,
        89: 15,
        111: 12,
        103: 12,
        39: 12,
        97: 12,
        0: 8,
    },
    "office_4": {
        10: 7,
        51: 6,
        52: 6,
        16: 15,
        18: 15,
        14: 5,
    }
}


def ids_to_nyu40():
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
    mapping = {}
    with open(os.path.join('data', 'scannetv2-labels.combined.tsv')) as tsvfile:
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


def nyu40_to_scannet21():
    reduce_map = {0: 0}
    for idx, cllist in enumerate([x.strip().split(',') for x in Path("data/scannet_to_reduced_scannet.csv").read_text().strip().splitlines()]):
        if cllist[1] != '':
            # reduce_map[idx + 1] = all_classes.index(cllist[1]) + 1
            reduce_map[idx + 1] = CLASSES_21.index(cllist[1]) + 1
        else:
            # reduce_map[idx + 1] = idx + 1
            reduce_map[idx + 1] = CLASSES_21.index(cllist[0]) + 1
    
    return reduce_map
    

def ids_to_scannet21():
    map40 = ids_to_nyu40()
    
    all_classes = []
    for cllist in [x.strip().split(',') for x in Path("data/scannet_to_reduced_scannet.csv").read_text().strip().splitlines()]:
        all_classes.append(cllist[0])
    
    reduce_map = nyu40_to_scannet21()
    
    # # original keys to 21 main values in range [0, 40]
    mapping = {k: v for k, v in map40.items()}

    # # original keys to 21 main values in range [0, 20]
    mapping = {k: reduce_map[v] for k, v in mapping.items()}
    
    max_id = max(mapping.keys())
    # Map relevant classes to {0,1,...,19}, and ignored classes to 255
    remapper = np.zeros(max_id+1, dtype=np.int64)
    for k, v in mapping.items():
        remapper[k] = v
    
    remapper = remapper - 1
    remapper[remapper==-1] = 255
    
    return remapper


if __name__ == '__main__':
    ids_to_scannet21()

