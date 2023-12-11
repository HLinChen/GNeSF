# Copyright (c) Facebook, Inc. and its affiliates.
import os
import numpy as np
import logging

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

# from 

logger = logging.getLogger(__name__)


COCO_CATEGORIES = [ # actual id in annotations: id - 1
    {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "person"},
    {"color": [119, 11, 32], "isthing": 1, "id": 2, "name": "bicycle"},
    {"color": [0, 0, 142], "isthing": 1, "id": 3, "name": "car"},
    {"color": [0, 0, 230], "isthing": 1, "id": 4, "name": "motorcycle"},
    {"color": [106, 0, 228], "isthing": 1, "id": 5, "name": "airplane"},
    {"color": [0, 60, 100], "isthing": 1, "id": 6, "name": "bus"},
    {"color": [0, 80, 100], "isthing": 1, "id": 7, "name": "train"},
    {"color": [0, 0, 70], "isthing": 1, "id": 8, "name": "truck"},
    {"color": [0, 0, 192], "isthing": 1, "id": 9, "name": "boat"},
    {"color": [250, 170, 30], "isthing": 1, "id": 10, "name": "traffic light"},
    {"color": [100, 170, 30], "isthing": 1, "id": 11, "name": "fire hydrant"},
    {"color": [220, 220, 0], "isthing": 1, "id": 13, "name": "stop sign"},
    {"color": [175, 116, 175], "isthing": 1, "id": 14, "name": "parking meter"},
    {"color": [250, 0, 30], "isthing": 1, "id": 15, "name": "bench"},
    {"color": [165, 42, 42], "isthing": 1, "id": 16, "name": "bird"},
    {"color": [255, 77, 255], "isthing": 1, "id": 17, "name": "cat"},
    {"color": [0, 226, 252], "isthing": 1, "id": 18, "name": "dog"},
    {"color": [182, 182, 255], "isthing": 1, "id": 19, "name": "horse"},
    {"color": [0, 82, 0], "isthing": 1, "id": 20, "name": "sheep"},
    {"color": [120, 166, 157], "isthing": 1, "id": 21, "name": "cow"},
    {"color": [110, 76, 0], "isthing": 1, "id": 22, "name": "elephant"},
    {"color": [174, 57, 255], "isthing": 1, "id": 23, "name": "bear"},
    {"color": [199, 100, 0], "isthing": 1, "id": 24, "name": "zebra"},
    {"color": [72, 0, 118], "isthing": 1, "id": 25, "name": "giraffe"},
    {"color": [255, 179, 240], "isthing": 1, "id": 27, "name": "backpack"},
    {"color": [0, 125, 92], "isthing": 1, "id": 28, "name": "umbrella"},
    {"color": [209, 0, 151], "isthing": 1, "id": 31, "name": "handbag"},
    {"color": [188, 208, 182], "isthing": 1, "id": 32, "name": "tie"},
    {"color": [0, 220, 176], "isthing": 1, "id": 33, "name": "suitcase"},
    {"color": [255, 99, 164], "isthing": 1, "id": 34, "name": "frisbee"},
    {"color": [92, 0, 73], "isthing": 1, "id": 35, "name": "skis"},
    {"color": [133, 129, 255], "isthing": 1, "id": 36, "name": "snowboard"},
    {"color": [78, 180, 255], "isthing": 1, "id": 37, "name": "sports ball"},
    {"color": [0, 228, 0], "isthing": 1, "id": 38, "name": "kite"},
    {"color": [174, 255, 243], "isthing": 1, "id": 39, "name": "baseball bat"},
    {"color": [45, 89, 255], "isthing": 1, "id": 40, "name": "baseball glove"},
    {"color": [134, 134, 103], "isthing": 1, "id": 41, "name": "skateboard"},
    {"color": [145, 148, 174], "isthing": 1, "id": 42, "name": "surfboard"},
    {"color": [255, 208, 186], "isthing": 1, "id": 43, "name": "tennis racket"},
    {"color": [197, 226, 255], "isthing": 1, "id": 44, "name": "bottle"},
    {"color": [171, 134, 1], "isthing": 1, "id": 46, "name": "wine glass"},
    {"color": [109, 63, 54], "isthing": 1, "id": 47, "name": "cup"},
    {"color": [207, 138, 255], "isthing": 1, "id": 48, "name": "fork"},
    {"color": [151, 0, 95], "isthing": 1, "id": 49, "name": "knife"},
    {"color": [9, 80, 61], "isthing": 1, "id": 50, "name": "spoon"},
    {"color": [84, 105, 51], "isthing": 1, "id": 51, "name": "bowl"},
    {"color": [74, 65, 105], "isthing": 1, "id": 52, "name": "banana"},
    {"color": [166, 196, 102], "isthing": 1, "id": 53, "name": "apple"},
    {"color": [208, 195, 210], "isthing": 1, "id": 54, "name": "sandwich"},
    {"color": [255, 109, 65], "isthing": 1, "id": 55, "name": "orange"},
    {"color": [0, 143, 149], "isthing": 1, "id": 56, "name": "broccoli"},
    {"color": [179, 0, 194], "isthing": 1, "id": 57, "name": "carrot"},
    {"color": [209, 99, 106], "isthing": 1, "id": 58, "name": "hot dog"},
    {"color": [5, 121, 0], "isthing": 1, "id": 59, "name": "pizza"},
    {"color": [227, 255, 205], "isthing": 1, "id": 60, "name": "donut"},
    {"color": [147, 186, 208], "isthing": 1, "id": 61, "name": "cake"},
    {"color": [153, 69, 1], "isthing": 1, "id": 62, "name": "chair"},
    {"color": [3, 95, 161], "isthing": 1, "id": 63, "name": "couch"},
    {"color": [163, 255, 0], "isthing": 1, "id": 64, "name": "potted plant"},
    {"color": [119, 0, 170], "isthing": 1, "id": 65, "name": "bed"},
    {"color": [0, 182, 199], "isthing": 1, "id": 67, "name": "dining table"},
    {"color": [0, 165, 120], "isthing": 1, "id": 70, "name": "toilet"},
    {"color": [183, 130, 88], "isthing": 1, "id": 72, "name": "tv"},
    {"color": [95, 32, 0], "isthing": 1, "id": 73, "name": "laptop"},
    {"color": [130, 114, 135], "isthing": 1, "id": 74, "name": "mouse"},
    {"color": [110, 129, 133], "isthing": 1, "id": 75, "name": "remote"},
    {"color": [166, 74, 118], "isthing": 1, "id": 76, "name": "keyboard"},
    {"color": [219, 142, 185], "isthing": 1, "id": 77, "name": "cell phone"},
    {"color": [79, 210, 114], "isthing": 1, "id": 78, "name": "microwave"},
    {"color": [178, 90, 62], "isthing": 1, "id": 79, "name": "oven"},
    {"color": [65, 70, 15], "isthing": 1, "id": 80, "name": "toaster"},
    {"color": [127, 167, 115], "isthing": 1, "id": 81, "name": "sink"},
    {"color": [59, 105, 106], "isthing": 1, "id": 82, "name": "refrigerator"},
    {"color": [142, 108, 45], "isthing": 1, "id": 84, "name": "book"},
    {"color": [196, 172, 0], "isthing": 1, "id": 85, "name": "clock"},
    {"color": [95, 54, 80], "isthing": 1, "id": 86, "name": "vase"},
    {"color": [128, 76, 255], "isthing": 1, "id": 87, "name": "scissors"},
    {"color": [201, 57, 1], "isthing": 1, "id": 88, "name": "teddy bear"},
    {"color": [246, 0, 122], "isthing": 1, "id": 89, "name": "hair drier"},
    {"color": [191, 162, 208], "isthing": 1, "id": 90, "name": "toothbrush"},
    {"id": 92, "name": "banner", "supercategory": "textile"},
    {"id": 93, "name": "blanket", "supercategory": "textile"},
    {"id": 94, "name": "branch", "supercategory": "plant"},
    {"id": 95, "name": "bridge", "supercategory": "building"},
    {"id": 96, "name": "building-other", "supercategory": "building"},
    {"id": 97, "name": "bush", "supercategory": "plant"},
    {"id": 98, "name": "cabinet", "supercategory": "furniture-stuff"},
    {"id": 99, "name": "cage", "supercategory": "structural"},
    {"id": 100, "name": "cardboard", "supercategory": "raw-material"},
    {"id": 101, "name": "carpet", "supercategory": "floor"},
    {"id": 102, "name": "ceiling-other", "supercategory": "ceiling"},
    {"id": 103, "name": "ceiling-tile", "supercategory": "ceiling"},
    {"id": 104, "name": "cloth", "supercategory": "textile"},
    {"id": 105, "name": "clothes", "supercategory": "textile"},
    {"id": 106, "name": "clouds", "supercategory": "sky"},
    {"id": 107, "name": "counter", "supercategory": "furniture-stuff"},
    {"id": 108, "name": "cupboard", "supercategory": "furniture-stuff"},
    {"id": 109, "name": "curtain", "supercategory": "textile"},
    {"id": 110, "name": "desk-stuff", "supercategory": "furniture-stuff"},
    {"id": 111, "name": "dirt", "supercategory": "ground"},
    {"id": 112, "name": "door-stuff", "supercategory": "furniture-stuff"},
    {"id": 113, "name": "fence", "supercategory": "structural"},
    {"id": 114, "name": "floor-marble", "supercategory": "floor"},
    {"id": 115, "name": "floor-other", "supercategory": "floor"},
    {"id": 116, "name": "floor-stone", "supercategory": "floor"},
    {"id": 117, "name": "floor-tile", "supercategory": "floor"},
    {"id": 118, "name": "floor-wood", "supercategory": "floor"},
    {"id": 119, "name": "flower", "supercategory": "plant"},
    {"id": 120, "name": "fog", "supercategory": "water"},
    {"id": 121, "name": "food-other", "supercategory": "food-stuff"},
    {"id": 122, "name": "fruit", "supercategory": "food-stuff"},
    {"id": 123, "name": "furniture-other", "supercategory": "furniture-stuff"},
    {"id": 124, "name": "grass", "supercategory": "plant"},
    {"id": 125, "name": "gravel", "supercategory": "ground"},
    {"id": 126, "name": "ground-other", "supercategory": "ground"},
    {"id": 127, "name": "hill", "supercategory": "solid"},
    {"id": 128, "name": "house", "supercategory": "building"},
    {"id": 129, "name": "leaves", "supercategory": "plant"},
    {"id": 130, "name": "light", "supercategory": "furniture-stuff"},
    {"id": 131, "name": "mat", "supercategory": "textile"},
    {"id": 132, "name": "metal", "supercategory": "raw-material"},
    {"id": 133, "name": "mirror-stuff", "supercategory": "furniture-stuff"},
    {"id": 134, "name": "moss", "supercategory": "plant"},
    {"id": 135, "name": "mountain", "supercategory": "solid"},
    {"id": 136, "name": "mud", "supercategory": "ground"},
    {"id": 137, "name": "napkin", "supercategory": "textile"},
    {"id": 138, "name": "net", "supercategory": "structural"},
    {"id": 139, "name": "paper", "supercategory": "raw-material"},
    {"id": 140, "name": "pavement", "supercategory": "ground"},
    {"id": 141, "name": "pillow", "supercategory": "textile"},
    {"id": 142, "name": "plant-other", "supercategory": "plant"},
    {"id": 143, "name": "plastic", "supercategory": "raw-material"},
    {"id": 144, "name": "platform", "supercategory": "ground"},
    {"id": 145, "name": "playingfield", "supercategory": "ground"},
    {"id": 146, "name": "railing", "supercategory": "structural"},
    {"id": 147, "name": "railroad", "supercategory": "ground"},
    {"id": 148, "name": "river", "supercategory": "water"},
    {"id": 149, "name": "road", "supercategory": "ground"},
    {"id": 150, "name": "rock", "supercategory": "solid"},
    {"id": 151, "name": "roof", "supercategory": "building"},
    {"id": 152, "name": "rug", "supercategory": "textile"},
    {"id": 153, "name": "salad", "supercategory": "food-stuff"},
    {"id": 154, "name": "sand", "supercategory": "ground"},
    {"id": 155, "name": "sea", "supercategory": "water"},
    {"id": 156, "name": "shelf", "supercategory": "furniture-stuff"},
    {"id": 157, "name": "sky-other", "supercategory": "sky"},
    {"id": 158, "name": "skyscraper", "supercategory": "building"},
    {"id": 159, "name": "snow", "supercategory": "ground"},
    {"id": 160, "name": "solid-other", "supercategory": "solid"},
    {"id": 161, "name": "stairs", "supercategory": "furniture-stuff"},
    {"id": 162, "name": "stone", "supercategory": "solid"},
    {"id": 163, "name": "straw", "supercategory": "plant"},
    {"id": 164, "name": "structural-other", "supercategory": "structural"},
    {"id": 165, "name": "table", "supercategory": "furniture-stuff"},
    {"id": 166, "name": "tent", "supercategory": "building"},
    {"id": 167, "name": "textile-other", "supercategory": "textile"},
    {"id": 168, "name": "towel", "supercategory": "textile"},
    {"id": 169, "name": "tree", "supercategory": "plant"},
    {"id": 170, "name": "vegetable", "supercategory": "food-stuff"},
    {"id": 171, "name": "wall-brick", "supercategory": "wall"},
    {"id": 172, "name": "wall-concrete", "supercategory": "wall"},
    {"id": 173, "name": "wall-other", "supercategory": "wall"},
    {"id": 174, "name": "wall-panel", "supercategory": "wall"},
    {"id": 175, "name": "wall-stone", "supercategory": "wall"},
    {"id": 176, "name": "wall-tile", "supercategory": "wall"},
    {"id": 177, "name": "wall-wood", "supercategory": "wall"},
    {"id": 178, "name": "water-other", "supercategory": "water"},
    {"id": 179, "name": "waterdrops", "supercategory": "water"},
    {"id": 180, "name": "window-blind", "supercategory": "window"},
    {"id": 181, "name": "window-other", "supercategory": "window"},
    {"id": 182, "name": "wood", "supercategory": "solid"},
]


CLASS_NAME = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'banner', 'blanket', 'branch', 'bridge', 'building-other', 'bush', 'cabinet', 'cage', 'cardboard', 'carpet', 'ceiling-other', 'ceiling-tile', 'cloth', 'clothes', 'clouds', 'counter', 'cupboard', 'curtain', 'desk-stuff', 'dirt', 'door-stuff', 'fence', 'floor-marble', 'floor-other', 'floor-stone', 'floor-tile', 'floor-wood', 'flower', 'fog', 'food-other', 'fruit', 'furniture-other', 'grass', 'gravel', 'ground-other', 'hill', 'house', 'leaves', 'light', 'mat', 'metal', 'mirror-stuff', 'moss', 'mountain', 'mud', 'napkin', 'net', 'paper', 'pavement', 'pillow', 'plant-other', 'plastic', 'platform', 'playingfield', 'railing', 'railroad', 'river', 'road', 'rock', 'roof', 'rug', 'salad', 'sand', 'sea', 'shelf', 'sky-other', 'skyscraper', 'snow', 'solid-other', 'stairs', 'stone', 'straw', 'structural-other', 'table', 'tent', 'textile-other', 'towel', 'tree', 'vegetable', 'wall-brick', 'wall-concrete', 'wall-other', 'wall-panel', 'wall-stone', 'wall-tile', 'wall-wood', 'water-other', 'waterdrops', 'window-blind', 'window-other', 'wood']


# COCO_CATEGORIES_30 = [
#     {"id": 171, "name": "wall-brick", "supercategory": "wall"},
#     {"id": 172, "name": "wall-concrete", "supercategory": "wall"},
#     {"id": 173, "name": "wall-other", "supercategory": "wall"},
#     {"id": 174, "name": "wall-panel", "supercategory": "wall"},
#     {"id": 175, "name": "wall-stone", "supercategory": "wall"},
#     {"id": 176, "name": "wall-tile", "supercategory": "wall"},
#     {"id": 177, "name": "wall-wood", "supercategory": "wall"},                  # 1
#     {"id": 114, "name": "floor-marble", "supercategory": "floor"},
#     {"id": 115, "name": "floor-other", "supercategory": "floor"},
#     {"id": 116, "name": "floor-stone", "supercategory": "floor"},
#     {"id": 117, "name": "floor-tile", "supercategory": "floor"},
#     {"id": 118, "name": "floor-wood", "supercategory": "floor"},                # 2
#     {"id": 98, "name": "cabinet", "supercategory": "furniture-stuff"},          # 3
#     {"color": [119, 0, 170], "isthing": 1, "id": 65, "name": "bed"},            # 4
#     {"color": [153, 69, 1], "isthing": 1, "id": 62, "name": "chair"},           # 5
#     # {"color": [250, 0, 30], "isthing": 1, "id": 15, "name": "bench"},
#     {"color": [3, 95, 161], "isthing": 1, "id": 63, "name": "couch"},           # 6
#     {"color": [0, 182, 199], "isthing": 1, "id": 67, "name": "dining table"},   
#     {"id": 165, "name": "table", "supercategory": "furniture-stuff"},           
#     {"id": 110, "name": "desk-stuff", "supercategory": "furniture-stuff"},      # 7
#     {"id": 112, "name": "door-stuff", "supercategory": "furniture-stuff"},      # 8
#     {"id": 180, "name": "window-blind", "supercategory": "window"},             
#     {"id": 181, "name": "window-other", "supercategory": "window"},             # 9
#     {"id": 156, "name": "shelf", "supercategory": "furniture-stuff"},           # 10
#     # picture
#     {"id": 107, "name": "counter", "supercategory": "furniture-stuff"},         # 11
#     {"id": 109, "name": "curtain", "supercategory": "textile"},                 # 12
#     {"color": [59, 105, 106], "isthing": 1, "id": 82, "name": "refrigerator"},  # 13
#     {"color": [0, 165, 120], "isthing": 1, "id": 70, "name": "toilet"},         # 14
#     {"color": [127, 167, 115], "isthing": 1, "id": 81, "name": "sink"},         # 15
#     # bathtub
    
#     {"id": 102, "name": "ceiling-other", "supercategory": "ceiling"},
#     {"id": 103, "name": "ceiling-tile", "supercategory": "ceiling"},            # 16
#     {"color": [183, 130, 88], "isthing": 1, "id": 72, "name": "tv"},            # 17
#     {"id": 141, "name": "pillow", "supercategory": "textile"},                  # 18
#     {"id": 133, "name": "mirror-stuff", "supercategory": "furniture-stuff"},    # 19
#     {"id": 101, "name": "carpet", "supercategory": "floor"},                    
#     {"id": 152, "name": "rug", "supercategory": "textile"},                     
#     {"id": 131, "name": "mat", "supercategory": "textile"},                     # 20
#     {"id": 130, "name": "light", "supercategory": "furniture-stuff"},           # 21
#     {"color": [142, 108, 45], "isthing": 1, "id": 84, "name": "book"},          # 22
#     {"id": 139, "name": "paper", "supercategory": "raw-material"},              # 23
#     {"id": 168, "name": "towel", "supercategory": "textile"},                   # 24
#     # box
#     # whiteboard
#     {"color": [209, 0, 151], "isthing": 1, "id": 31, "name": "handbag"},        
#     {"color": [255, 179, 240], "isthing": 1, "id": 27, "name": "backpack"},     # 26
    
    
    
#     # {"color": [0, 220, 176], "isthing": 1, "id": 33, "name": "suitcase"},
    
#     # {"color": [0, 125, 92], "isthing": 1, "id": 28, "name": "umbrella"},
#     # {"color": [188, 208, 182], "isthing": 1, "id": 32, "name": "tie"},
#     # {"color": [255, 99, 164], "isthing": 1, "id": 34, "name": "frisbee"},
#     # {"color": [92, 0, 73], "isthing": 1, "id": 35, "name": "skis"},
#     # {"color": [133, 129, 255], "isthing": 1, "id": 36, "name": "snowboard"},
#     # {"color": [78, 180, 255], "isthing": 1, "id": 37, "name": "sports ball"},
#     # {"color": [0, 228, 0], "isthing": 1, "id": 38, "name": "kite"},
#     # {"color": [174, 255, 243], "isthing": 1, "id": 39, "name": "baseball bat"},
#     # {"color": [45, 89, 255], "isthing": 1, "id": 40, "name": "baseball glove"},
#     # {"color": [134, 134, 103], "isthing": 1, "id": 41, "name": "skateboard"},
#     # {"color": [145, 148, 174], "isthing": 1, "id": 42, "name": "surfboard"},
#     # {"color": [255, 208, 186], "isthing": 1, "id": 43, "name": "tennis racket"},
#     # {"color": [197, 226, 255], "isthing": 1, "id": 44, "name": "bottle"},
#     # {"color": [171, 134, 1], "isthing": 1, "id": 46, "name": "wine glass"},
#     # {"color": [109, 63, 54], "isthing": 1, "id": 47, "name": "cup"},
#     # {"color": [207, 138, 255], "isthing": 1, "id": 48, "name": "fork"},
#     # {"color": [151, 0, 95], "isthing": 1, "id": 49, "name": "knife"},
#     # {"color": [9, 80, 61], "isthing": 1, "id": 50, "name": "spoon"},
#     # {"color": [84, 105, 51], "isthing": 1, "id": 51, "name": "bowl"},
#     # {"color": [74, 65, 105], "isthing": 1, "id": 52, "name": "banana"},
#     # {"color": [166, 196, 102], "isthing": 1, "id": 53, "name": "apple"},
#     # {"color": [208, 195, 210], "isthing": 1, "id": 54, "name": "sandwich"},
#     # {"color": [255, 109, 65], "isthing": 1, "id": 55, "name": "orange"},
#     # {"color": [0, 143, 149], "isthing": 1, "id": 56, "name": "broccoli"},
#     # {"color": [179, 0, 194], "isthing": 1, "id": 57, "name": "carrot"},
#     # {"color": [209, 99, 106], "isthing": 1, "id": 58, "name": "hot dog"},
#     # {"color": [5, 121, 0], "isthing": 1, "id": 59, "name": "pizza"},
#     # {"color": [227, 255, 205], "isthing": 1, "id": 60, "name": "donut"},
#     # {"color": [147, 186, 208], "isthing": 1, "id": 61, "name": "cake"},
#     # {"color": [163, 255, 0], "isthing": 1, "id": 64, "name": "potted plant"},
#     # {"color": [95, 32, 0], "isthing": 1, "id": 73, "name": "laptop"},
#     # {"color": [130, 114, 135], "isthing": 1, "id": 74, "name": "mouse"},
#     # {"color": [110, 129, 133], "isthing": 1, "id": 75, "name": "remote"},
#     # {"color": [166, 74, 118], "isthing": 1, "id": 76, "name": "keyboard"},
#     # {"color": [219, 142, 185], "isthing": 1, "id": 77, "name": "cell phone"},
#     # {"color": [79, 210, 114], "isthing": 1, "id": 78, "name": "microwave"},
#     # {"color": [178, 90, 62], "isthing": 1, "id": 79, "name": "oven"},
#     # {"color": [65, 70, 15], "isthing": 1, "id": 80, "name": "toaster"},
#     # {"color": [196, 172, 0], "isthing": 1, "id": 85, "name": "clock"},
#     # {"color": [95, 54, 80], "isthing": 1, "id": 86, "name": "vase"},
#     # {"color": [128, 76, 255], "isthing": 1, "id": 87, "name": "scissors"},
#     # {"color": [201, 57, 1], "isthing": 1, "id": 88, "name": "teddy bear"},
#     # {"color": [246, 0, 122], "isthing": 1, "id": 89, "name": "hair drier"},
#     # {"color": [191, 162, 208], "isthing": 1, "id": 90, "name": "toothbrush"},
#     # {"id": 92, "name": "banner", "supercategory": "textile"},
#     # {"id": 93, "name": "blanket", "supercategory": "textile"},
#     # {"id": 100, "name": "cardboard", "supercategory": "raw-material"},
#     # {"id": 104, "name": "cloth", "supercategory": "textile"},
#     # {"id": 105, "name": "clothes", "supercategory": "textile"},
#     # {"id": 108, "name": "cupboard", "supercategory": "furniture-stuff"},
#     # {"id": 119, "name": "flower", "supercategory": "plant"},
#     # {"id": 123, "name": "furniture-other", "supercategory": "furniture-stuff"},
#     # {"id": 137, "name": "napkin", "supercategory": "textile"},
#     # {"id": 138, "name": "net", "supercategory": "structural"},
#     # {"id": 146, "name": "railing", "supercategory": "structural"},
#     # {"id": 161, "name": "stairs", "supercategory": "furniture-stuff"},
#     # {"id": 164, "name": "structural-other", "supercategory": "structural"},
#     # {"id": 121, "name": "food-other", "supercategory": "food-stuff"},
#     # {"id": 122, "name": "fruit", "supercategory": "food-stuff"},
#     # {"id": 170, "name": "vegetable", "supercategory": "food-stuff"},
# ]


COCO_CATEGORIES_SCANNET = [
    # {"color": [0, 0, 0], "id": 0, "isthing": 1, "name": "other"},
    {"color": [174, 199, 232], "id": 1, "isthing": 0, "name": "wall"},
    {"color": [152, 223, 138], "id": 2, "isthing": 0, "name": "floor"},
    {"color": [31, 119, 180], "id": 3, "isthing": 1, "name": "cabinet"},
    {"color": [255, 187, 120], "id": 4, "isthing": 1, "name": "bed"},
    {
        "color": [188, 189, 34],
        "id": 5,
        "isthing": 1,
        "name": "chair",
    },
    {"color": [140, 86, 75], "id": 6, "isthing": 1, "name": "sofa"},
    {
        "color": [255, 152, 150],
        "id": 7,
        "isthing": 1,
        "name": "desk, table", # , pool table, billiard table, snooker table , desk
    },
    {
        "color": [214, 39, 40], 
        "id": 8, 
        "isthing": 1, 
        "name": "door"
    },
    {"color": [197, 176, 213], "id": 9, "isthing": 1, "name": "window"},
    {"color": [148, 103, 189], "id": 10, "isthing": 1, "name": "bookshelf, shleves"},
    # {"color": [196, 156, 148], "id": 11, "isthing": 1, "name": "picture"},
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
    # {"color": [227, 119, 194], "id": 17, "isthing": 1, "name": "bathtub"},
    
    {"color": [78, 71, 183], "id": 16, "isthing": 0, "name": "ceiling"},
    {"color": [91, 163, 138], "id": 17, "isthing": 1, "name": "television"},
    {"color": [202, 185, 52], "id": 18, "isthing": 1, "name": "pillow"},
    {"color": [51, 176, 203], "id": 19, "isthing": 1, "name": "mirror"},
    {"color": [200, 54, 131], "id": 20, "isthing": 0, "name": "floor mat"}, # floormat ?
    {"color": [96, 207, 209], "id": 21, "isthing": 1, "name": "lamp"},
    
    {"color": [172, 114, 82], "id": 22, "isthing": 1, "name": "books"},
    {"color": [153, 98, 156], "id": 23, "isthing": 1, "name": "paper"},
    {"color": [140, 153, 101], "id": 24, "isthing": 1, "name": "towel"},
    # {"color": [100, 125, 154], "id": 27, "isthing": 1, "name": "box"},
    # {"color": [178, 127, 135], "id": 28, "isthing": 1, "name": "whiteboard"},
    {"color": [213, 92, 176], "id": 25, "isthing": 1, "name": "bag"},
    # {"color": [146, 111, 194], "id": 30, "isthing": 1, "name": "night stand"},
    
    
    
    # {"id": 171, "name": "wall-brick", "supercategory": "wall"},
    # {"id": 172, "name": "wall-concrete", "supercategory": "wall"},
    # {"id": 173, "name": "wall-other", "supercategory": "wall"},
    # {"id": 174, "name": "wall-panel", "supercategory": "wall"},
    # {"id": 175, "name": "wall-stone", "supercategory": "wall"},
    # {"id": 176, "name": "wall-tile", "supercategory": "wall"},
    # {"id": 177, "name": "wall-wood", "supercategory": "wall"},                  # 1
    # {"id": 114, "name": "floor-marble", "supercategory": "floor"},
    # {"id": 115, "name": "floor-other", "supercategory": "floor"},
    # {"id": 116, "name": "floor-stone", "supercategory": "floor"},
    # {"id": 117, "name": "floor-tile", "supercategory": "floor"},
    # {"id": 118, "name": "floor-wood", "supercategory": "floor"},                # 2
    # {"id": 98, "name": "cabinet", "supercategory": "furniture-stuff"},          # 3
    # {"color": [119, 0, 170], "isthing": 1, "id": 65, "name": "bed"},            # 4
    # {"color": [153, 69, 1], "isthing": 1, "id": 62, "name": "chair"},           # 5
    # # {"color": [250, 0, 30], "isthing": 1, "id": 15, "name": "bench"},
    # {"color": [3, 95, 161], "isthing": 1, "id": 63, "name": "couch"},           # 6
    # {"color": [0, 182, 199], "isthing": 1, "id": 67, "name": "dining table"},   
    # {"id": 165, "name": "table", "supercategory": "furniture-stuff"},           
    # {"id": 110, "name": "desk-stuff", "supercategory": "furniture-stuff"},      # 7
    # {"id": 112, "name": "door-stuff", "supercategory": "furniture-stuff"},      # 8
    # {"id": 180, "name": "window-blind", "supercategory": "window"},             
    # {"id": 181, "name": "window-other", "supercategory": "window"},             # 9
    # {"id": 156, "name": "shelf", "supercategory": "furniture-stuff"},           # 10
    # # picture
    # {"id": 107, "name": "counter", "supercategory": "furniture-stuff"},         # 11
    # {"id": 109, "name": "curtain", "supercategory": "textile"},                 # 12
    # {"color": [59, 105, 106], "isthing": 1, "id": 82, "name": "refrigerator"},  # 13
    # {"color": [0, 165, 120], "isthing": 1, "id": 70, "name": "toilet"},         # 14
    # {"color": [127, 167, 115], "isthing": 1, "id": 81, "name": "sink"},         # 15
    # # bathtub
    
    # {"id": 102, "name": "ceiling-other", "supercategory": "ceiling"},
    # {"id": 103, "name": "ceiling-tile", "supercategory": "ceiling"},            # 16
    # {"color": [183, 130, 88], "isthing": 1, "id": 72, "name": "tv"},            # 17
    # {"id": 141, "name": "pillow", "supercategory": "textile"},                  # 18
    # {"id": 133, "name": "mirror-stuff", "supercategory": "furniture-stuff"},    # 19
    # {"id": 101, "name": "carpet", "supercategory": "floor"},                    
    # {"id": 152, "name": "rug", "supercategory": "textile"},                     
    # {"id": 131, "name": "mat", "supercategory": "textile"},                     # 20
    # {"id": 130, "name": "light", "supercategory": "furniture-stuff"},           # 21
    # {"color": [142, 108, 45], "isthing": 1, "id": 84, "name": "book"},          # 22
    # {"id": 139, "name": "paper", "supercategory": "raw-material"},              # 23
    # {"id": 168, "name": "towel", "supercategory": "textile"},                   # 24
    # # box
    # # whiteboard
    # {"color": [209, 0, 151], "isthing": 1, "id": 31, "name": "handbag"},        
    # {"color": [255, 179, 240], "isthing": 1, "id": 27, "name": "backpack"},     # 25
    
    
    
    # # {"color": [0, 220, 176], "isthing": 1, "id": 33, "name": "suitcase"},
    
    # # {"color": [0, 125, 92], "isthing": 1, "id": 28, "name": "umbrella"},
    # # {"color": [188, 208, 182], "isthing": 1, "id": 32, "name": "tie"},
    # # {"color": [255, 99, 164], "isthing": 1, "id": 34, "name": "frisbee"},
    # # {"color": [92, 0, 73], "isthing": 1, "id": 35, "name": "skis"},
    # # {"color": [133, 129, 255], "isthing": 1, "id": 36, "name": "snowboard"},
    # # {"color": [78, 180, 255], "isthing": 1, "id": 37, "name": "sports ball"},
    # # {"color": [0, 228, 0], "isthing": 1, "id": 38, "name": "kite"},
    # # {"color": [174, 255, 243], "isthing": 1, "id": 39, "name": "baseball bat"},
    # # {"color": [45, 89, 255], "isthing": 1, "id": 40, "name": "baseball glove"},
    # # {"color": [134, 134, 103], "isthing": 1, "id": 41, "name": "skateboard"},
    # # {"color": [145, 148, 174], "isthing": 1, "id": 42, "name": "surfboard"},
    # # {"color": [255, 208, 186], "isthing": 1, "id": 43, "name": "tennis racket"},
    # # {"color": [197, 226, 255], "isthing": 1, "id": 44, "name": "bottle"},
    # # {"color": [171, 134, 1], "isthing": 1, "id": 46, "name": "wine glass"},
    # # {"color": [109, 63, 54], "isthing": 1, "id": 47, "name": "cup"},
    # # {"color": [207, 138, 255], "isthing": 1, "id": 48, "name": "fork"},
    # # {"color": [151, 0, 95], "isthing": 1, "id": 49, "name": "knife"},
    # # {"color": [9, 80, 61], "isthing": 1, "id": 50, "name": "spoon"},
    # # {"color": [84, 105, 51], "isthing": 1, "id": 51, "name": "bowl"},
    # # {"color": [74, 65, 105], "isthing": 1, "id": 52, "name": "banana"},
    # # {"color": [166, 196, 102], "isthing": 1, "id": 53, "name": "apple"},
    # # {"color": [208, 195, 210], "isthing": 1, "id": 54, "name": "sandwich"},
    # # {"color": [255, 109, 65], "isthing": 1, "id": 55, "name": "orange"},
    # # {"color": [0, 143, 149], "isthing": 1, "id": 56, "name": "broccoli"},
    # # {"color": [179, 0, 194], "isthing": 1, "id": 57, "name": "carrot"},
    # # {"color": [209, 99, 106], "isthing": 1, "id": 58, "name": "hot dog"},
    # # {"color": [5, 121, 0], "isthing": 1, "id": 59, "name": "pizza"},
    # # {"color": [227, 255, 205], "isthing": 1, "id": 60, "name": "donut"},
    # # {"color": [147, 186, 208], "isthing": 1, "id": 61, "name": "cake"},
    # # {"color": [163, 255, 0], "isthing": 1, "id": 64, "name": "potted plant"},
    # # {"color": [95, 32, 0], "isthing": 1, "id": 73, "name": "laptop"},
    # # {"color": [130, 114, 135], "isthing": 1, "id": 74, "name": "mouse"},
    # # {"color": [110, 129, 133], "isthing": 1, "id": 75, "name": "remote"},
    # # {"color": [166, 74, 118], "isthing": 1, "id": 76, "name": "keyboard"},
    # # {"color": [219, 142, 185], "isthing": 1, "id": 77, "name": "cell phone"},
    # # {"color": [79, 210, 114], "isthing": 1, "id": 78, "name": "microwave"},
    # # {"color": [178, 90, 62], "isthing": 1, "id": 79, "name": "oven"},
    # # {"color": [65, 70, 15], "isthing": 1, "id": 80, "name": "toaster"},
    # # {"color": [196, 172, 0], "isthing": 1, "id": 85, "name": "clock"},
    # # {"color": [95, 54, 80], "isthing": 1, "id": 86, "name": "vase"},
    # # {"color": [128, 76, 255], "isthing": 1, "id": 87, "name": "scissors"},
    # # {"color": [201, 57, 1], "isthing": 1, "id": 88, "name": "teddy bear"},
    # # {"color": [246, 0, 122], "isthing": 1, "id": 89, "name": "hair drier"},
    # # {"color": [191, 162, 208], "isthing": 1, "id": 90, "name": "toothbrush"},
    # # {"id": 92, "name": "banner", "supercategory": "textile"},
    # # {"id": 93, "name": "blanket", "supercategory": "textile"},
    # # {"id": 100, "name": "cardboard", "supercategory": "raw-material"},
    # # {"id": 104, "name": "cloth", "supercategory": "textile"},
    # # {"id": 105, "name": "clothes", "supercategory": "textile"},
    # # {"id": 108, "name": "cupboard", "supercategory": "furniture-stuff"},
    # # {"id": 119, "name": "flower", "supercategory": "plant"},
    # # {"id": 123, "name": "furniture-other", "supercategory": "furniture-stuff"},
    # # {"id": 137, "name": "napkin", "supercategory": "textile"},
    # # {"id": 138, "name": "net", "supercategory": "structural"},
    # # {"id": 146, "name": "railing", "supercategory": "structural"},
    # # {"id": 161, "name": "stairs", "supercategory": "furniture-stuff"},
    # # {"id": 164, "name": "structural-other", "supercategory": "structural"},
    # # {"id": 121, "name": "food-other", "supercategory": "food-stuff"},
    # # {"id": 122, "name": "fruit", "supercategory": "food-stuff"},
    # # {"id": 170, "name": "vegetable", "supercategory": "food-stuff"},
]


COCO_CATEGORIES_SCANNET_21 = [
    # {"color": [0, 0, 0], "id": 0, "isthing": 1, "name": "other"},
    {"color": [174, 199, 232], "id": 1, "isthing": 0, "name": "wall"},
    {"color": [152, 223, 138], "id": 2, "isthing": 0, "name": "floor"}, # , floor mat
    {"color": [31, 119, 180], "id": 3, "isthing": 1, "name": "cabinet"},
    {"color": [255, 187, 120], "id": 4, "isthing": 1, "name": "bed"},
    {
        "color": [188, 189, 34],
        "id": 5,
        "isthing": 1,
        "name": "chair",
    },
    {"color": [140, 86, 75], "id": 6, "isthing": 1, "name": "sofa"},
    {
        "color": [255, 152, 150],
        "id": 7,
        "isthing": 1,
        "name": "table", # , pool table, billiard table, snooker table , desk desk, 
    },
    {
        "color": [214, 39, 40], 
        "id": 8, 
        "isthing": 1, 
        "name": "door"
    },
    {"color": [197, 176, 213], "id": 9, "isthing": 1, "name": "window"},
    {
        "color": [23, 190, 207], 
        "id": 10, 
        "isthing": 1, 
        "name": "counter"
    },
    {"color": [148, 103, 189], "id": 11, "isthing": 1, "name": "shelves"}, # , shleves
    {"color": [219, 219, 141], "id": 12, "isthing": 0, "name": "curtain"}, # blinds ?, shower curtain
    # {"color": [196, 156, 148], "id": 11, "isthing": 1, "name": "picture"},
    {"color": [78, 71, 183], "id": 13, "isthing": 0, "name": "ceiling"},
    {"color": [255, 127, 14], "id": 14, "isthing": 1, "name": "refridgerator"},
    {"color": [158, 218, 229], "id": 15, "isthing": 0, "name": "television"}, # blinds ? shower curtain
    {"color": [158, 218, 229], "id": 16, "isthing": 0, "name": "person"}, # blinds ? shower curtain
    # {"color": [247, 182, 210], "id": 13, "isthing": 0, "name": "desk"}, # blinds ?, shower curtain
    {
        "color": [44, 160, 44],
        "id": 17,
        "isthing": 1,
        "name": "toilet",
    },
    {"color": [112, 128, 144], "id": 18, "isthing": 1, "name": "sink"},
    {"color": [112, 128, 144], "id": 19, "isthing": 1, "name": "lamp"},
    {"color": [213, 92, 176], "id": 20, "isthing": 1, "name": "bag"},
    {"color": [213, 92, 176], "id": 21, "isthing": 1, "name": "otherprop"},
    
    # {"color": [227, 119, 194], "id": 17, "isthing": 1, "name": "bathtub"},
    
    # {"color": [78, 71, 183], "id": 18, "isthing": 0, "name": "ceiling"},
    # {"color": [91, 163, 138], "id": 17, "isthing": 1, "name": "television"},
    # {"color": [202, 185, 52], "id": 18, "isthing": 1, "name": "pillow"},
    # {"color": [51, 176, 203], "id": 19, "isthing": 1, "name": "mirror"},
    # {"color": [200, 54, 131], "id": 20, "isthing": 0, "name": "floor mat"}, # floormat ?
    # {"color": [96, 207, 209], "id": 21, "isthing": 1, "name": "lamp"},
    
    # {"color": [172, 114, 82], "id": 22, "isthing": 1, "name": "books"},
    # {"color": [153, 98, 156], "id": 23, "isthing": 1, "name": "paper"},
    # {"color": [140, 153, 101], "id": 24, "isthing": 1, "name": "towel"},
    # # {"color": [100, 125, 154], "id": 27, "isthing": 1, "name": "box"},
    # # {"color": [178, 127, 135], "id": 28, "isthing": 1, "name": "whiteboard"},
    # # {"color": [146, 111, 194], "id": 30, "isthing": 1, "name": "night stand"},
    
]


COCO_CATEGORIES_SCANNET_18 = [
    # {"color": [0, 0, 0], "id": 0, "isthing": 1, "name": "other"},
    {"color": [174, 199, 232], "id": 1, "isthing": 0, "name": "wall"},
    {"color": [152, 223, 138], "id": 2, "isthing": 0, "name": "floor, floor mat"}, # 
    {"color": [31, 119, 180], "id": 3, "isthing": 1, "name": "cabinet"},
    {"color": [255, 187, 120], "id": 4, "isthing": 1, "name": "bed"},
    {
        "color": [188, 189, 34],
        "id": 5,
        "isthing": 1,
        "name": "chair",
    },
    {"color": [140, 86, 75], "id": 6, "isthing": 1, "name": "sofa"},
    {
        "color": [255, 152, 150],
        "id": 7,
        "isthing": 1,
        "name": "table", # , pool table, billiard table, snooker table , desk desk, 
    },
    {
        "color": [214, 39, 40], 
        "id": 8, 
        "isthing": 1, 
        "name": "door"
    },
    {"color": [197, 176, 213], "id": 9, "isthing": 1, "name": "window"},
    {"color": [148, 103, 189], "id": 10, "isthing": 1, "name": "shelf"}, # , shleves
    # {"color": [196, 156, 148], "id": 11, "isthing": 1, "name": "picture"},
    {"color": [78, 71, 183], "id": 11, "isthing": 0, "name": "ceiling"},
    {
        "color": [23, 190, 207], 
        "id": 12, 
        "isthing": 1, 
        "name": "counter"
    },
    {"color": [247, 182, 210], "id": 13, "isthing": 0, "name": "desk"}, # blinds ?, shower curtain
    {"color": [219, 219, 141], "id": 14, "isthing": 0, "name": "curtain"}, # blinds ?, shower curtain
    {"color": [255, 127, 14], "id": 15, "isthing": 1, "name": "refridgerator"},
    {"color": [158, 218, 229], "id": 16, "isthing": 0, "name": "television"}, # blinds ? shower curtain
    {
        "color": [44, 160, 44],
        "id": 17,
        "isthing": 1,
        "name": "toilet",
    },
    {"color": [112, 128, 144], "id": 18, "isthing": 1, "name": "sink"},
    # {"color": [227, 119, 194], "id": 17, "isthing": 1, "name": "bathtub"},
    
    # {"color": [78, 71, 183], "id": 18, "isthing": 0, "name": "ceiling"},
    # {"color": [91, 163, 138], "id": 17, "isthing": 1, "name": "television"},
    # {"color": [202, 185, 52], "id": 18, "isthing": 1, "name": "pillow"},
    # {"color": [51, 176, 203], "id": 19, "isthing": 1, "name": "mirror"},
    # {"color": [200, 54, 131], "id": 20, "isthing": 0, "name": "floor mat"}, # floormat ?
    # {"color": [96, 207, 209], "id": 21, "isthing": 1, "name": "lamp"},
    
    # {"color": [172, 114, 82], "id": 22, "isthing": 1, "name": "books"},
    # {"color": [153, 98, 156], "id": 23, "isthing": 1, "name": "paper"},
    # {"color": [140, 153, 101], "id": 24, "isthing": 1, "name": "towel"},
    # # {"color": [100, 125, 154], "id": 27, "isthing": 1, "name": "box"},
    # # {"color": [178, 127, 135], "id": 28, "isthing": 1, "name": "whiteboard"},
    # {"color": [213, 92, 176], "id": 25, "isthing": 1, "name": "bag"},
    # # {"color": [146, 111, 194], "id": 30, "isthing": 1, "name": "night stand"},
    
]


COCO_CATEGORIES_SCANNET_18 = [
    # {"color": [0, 0, 0], "id": 0, "isthing": 1, "name": "other"},
    {"color": [174, 199, 232], "id": 1, "isthing": 0, "name": "wall"},
    {"color": [152, 223, 138], "id": 2, "isthing": 0, "name": "floor, floor mat"}, # 
    {"color": [31, 119, 180], "id": 3, "isthing": 1, "name": "cabinet"},
    {"color": [255, 187, 120], "id": 4, "isthing": 1, "name": "bed"},
    {
        "color": [188, 189, 34],
        "id": 5,
        "isthing": 1,
        "name": "chair",
    },
    {"color": [140, 86, 75], "id": 6, "isthing": 1, "name": "sofa"},
    {
        "color": [255, 152, 150],
        "id": 7,
        "isthing": 1,
        "name": "table", # , pool table, billiard table, snooker table , desk desk, 
    },
    {
        "color": [214, 39, 40], 
        "id": 8, 
        "isthing": 1, 
        "name": "door"
    },
    {"color": [197, 176, 213], "id": 9, "isthing": 1, "name": "window"},
    {"color": [148, 103, 189], "id": 10, "isthing": 1, "name": "shelf"}, # , shleves ?
    # {"color": [196, 156, 148], "id": 11, "isthing": 1, "name": "picture"},
    {"color": [78, 71, 183], "id": 11, "isthing": 0, "name": "ceiling"},    # ?
    {
        "color": [23, 190, 207], 
        "id": 12, 
        "isthing": 1, 
        "name": "counter"
    },
    {"color": [247, 182, 210], "id": 13, "isthing": 0, "name": "desk"},
    {"color": [219, 219, 141], "id": 14, "isthing": 0, "name": "curtain"},
    {"color": [255, 127, 14], "id": 15, "isthing": 1, "name": "refridgerator"},
    {"color": [158, 218, 229], "id": 16, "isthing": 0, "name": "television"}, # ?
    {
        "color": [44, 160, 44],
        "id": 17,
        "isthing": 1,
        "name": "toilet",
    },
    {"color": [112, 128, 144], "id": 18, "isthing": 1, "name": "sink"},
    # {"color": [227, 119, 194], "id": 17, "isthing": 1, "name": "bathtub"},
    
    # {"color": [78, 71, 183], "id": 18, "isthing": 0, "name": "ceiling"},
    # {"color": [91, 163, 138], "id": 17, "isthing": 1, "name": "television"},
    # {"color": [202, 185, 52], "id": 18, "isthing": 1, "name": "pillow"},
    # {"color": [51, 176, 203], "id": 19, "isthing": 1, "name": "mirror"},
    # {"color": [200, 54, 131], "id": 20, "isthing": 0, "name": "floor mat"}, # floormat ?
    # {"color": [96, 207, 209], "id": 21, "isthing": 1, "name": "lamp"},
    
    # {"color": [172, 114, 82], "id": 22, "isthing": 1, "name": "books"},
    # {"color": [153, 98, 156], "id": 23, "isthing": 1, "name": "paper"},
    # {"color": [140, 153, 101], "id": 24, "isthing": 1, "name": "towel"},
    # # {"color": [100, 125, 154], "id": 27, "isthing": 1, "name": "box"},
    # # {"color": [178, 127, 135], "id": 28, "isthing": 1, "name": "whiteboard"},
    # {"color": [213, 92, 176], "id": 25, "isthing": 1, "name": "bag"},
    # # {"color": [146, 111, 194], "id": 30, "isthing": 1, "name": "night stand"},
    
]


COCO_CATEGORIES_SCANNET_COLORS = [k["color"] for k in COCO_CATEGORIES_SCANNET]


COCO_CATEGORIES_SCANNET_COLORS_18 = [k["color"] for k in COCO_CATEGORIES_SCANNET_18]

COCO_CATEGORIES_SCANNET_COLORS_21 = [k["color"] for k in COCO_CATEGORIES_SCANNET_21]

def _get_coco_stuff_meta():
    # Id 0 is reserved for ignore_label, we change ignore_label for 0
    # to 255 in our pre-processing.
    stuff_ids = [k["id"] for k in COCO_CATEGORIES_SCANNET]
    # stuff_ids = [k["id"] - 1 for k in COCO_CATEGORIES]
    # stuff_ids = [k["id"] - 1 for k in COCO_CATEGORIES]
    # assert len(stuff_ids) == 171, len(stuff_ids)

    # For semantic segmentation, this mapping maps from contiguous stuff id
    # (in [0, 91], used in models) to ids in the dataset (used for processing results)
    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}
    stuff_classes = [k["name"] for k in COCO_CATEGORIES]

    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
    }
    return ret


def _get_coco_stuff_meta_scannet():
    # Id 0 is reserved for ignore_label, we change ignore_label for 0
    # to 255 in our pre-processing.
    # stuff_ids = [k["id"] for k in COCO_CATEGORIES_SCANNET]
    stuff_ids = [k["id"] - 1 for k in COCO_CATEGORIES_SCANNET]
    # stuff_ids = [k["id"] - 1 for k in COCO_CATEGORIES]
    # assert len(stuff_ids) == 171, len(stuff_ids)

    # For semantic segmentation, this mapping maps from contiguous stuff id
    # (in [0, 91], used in models) to ids in the dataset (used for processing results)
    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}
    stuff_classes = [k["name"] for k in COCO_CATEGORIES_SCANNET]

    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
    }
    return ret


def _get_coco_stuff_meta_scannet_18():
    # Id 0 is reserved for ignore_label, we change ignore_label for 0
    # to 255 in our pre-processing.
    # stuff_ids = [k["id"] for k in COCO_CATEGORIES_SCANNET]
    stuff_ids = [k["id"] - 1 for k in COCO_CATEGORIES_SCANNET_18]
    # stuff_ids = [k["id"] - 1 for k in COCO_CATEGORIES]
    # assert len(stuff_ids) == 171, len(stuff_ids)

    # For semantic segmentation, this mapping maps from contiguous stuff id
    # (in [0, 91], used in models) to ids in the dataset (used for processing results)
    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}
    stuff_classes = [k["name"] for k in COCO_CATEGORIES_SCANNET_18]

    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
    }
    return ret


def _get_coco_stuff_meta_scannet_21():
    # Id 0 is reserved for ignore_label, we change ignore_label for 0
    # to 255 in our pre-processing.
    # stuff_ids = [k["id"] for k in COCO_CATEGORIES_SCANNET]
    stuff_ids = [k["id"] - 1 for k in COCO_CATEGORIES_SCANNET_21]
    # stuff_ids = [k["id"] - 1 for k in COCO_CATEGORIES]
    # assert len(stuff_ids) == 171, len(stuff_ids)

    # For semantic segmentation, this mapping maps from contiguous stuff id
    # (in [0, 91], used in models) to ids in the dataset (used for processing results)
    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}
    stuff_classes = [k["name"] for k in COCO_CATEGORIES_SCANNET_21]

    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
    }
    return ret


def register_all_coco_stuff_164k(root):
    root = os.path.join(root, "coco_stuff") # _164k
    meta = _get_coco_stuff_meta()
    for name, image_dirname, sem_seg_dirname in [
        # ("train", "images_detectron2/train", "annotations_detectron2/train"),
        # ("test", "images_detectron2/test", "annotations_detectron2/test"),
        ("train", "images/train2017", "annotations/train2017"),
        ("test", "images/val2017", "annotations/val2017"),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        name = f"coco_2017_{name}_stuff_164k_sem_seg"
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="jpg")
        )
        MetadataCatalog.get(name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
            # stuff_colors=COCO_CATEGORIES_SCANNET_COLORS[:],
            **meta,
        )


def register_all_coco_stuff_164k_scannet(root):
    root = os.path.join(root, "coco_stuff") # _164k
    meta = _get_coco_stuff_meta_scannet()
    for name, image_dirname, sem_seg_dirname in [
        # ("train", "images_detectron2/train", "annotations_detectron2/train"),
        # ("test", "images_detectron2/test", "annotations_detectron2/test"),
        ("train", "images/train2017", "annotations/train2017"),
        ("test", "images/val2017", "annotations/val2017"),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        name = f"coco_2017_{name}_stuff_164k_sem_seg_scannet"
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="jpg")
        )
        MetadataCatalog.get(name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
            stuff_colors=COCO_CATEGORIES_SCANNET_COLORS[:],
            **meta,
        )


def register_all_coco_stuff_164k_scannet_18(root):
    root = os.path.join(root, "coco_stuff") # _164k
    meta = _get_coco_stuff_meta_scannet_18()
    for name, image_dirname, sem_seg_dirname in [
        # ("train", "images_detectron2/train", "annotations_detectron2/train"),
        # ("test", "images_detectron2/test", "annotations_detectron2/test"),
        ("train", "images/train2017", "annotations_18/train2017"),
        ("test", "images/val2017", "annotations_18/val2017"),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        name = f"coco_2017_{name}_stuff_164k_sem_seg_scannet_18"
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="jpg")
        )
        MetadataCatalog.get(name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
            stuff_colors=COCO_CATEGORIES_SCANNET_COLORS_18[:],
            **meta,
        )


def register_all_coco_stuff_164k_scannet_21(root):
    root = os.path.join(root, "coco_stuff") # _164k
    meta = _get_coco_stuff_meta_scannet_21()
    for name, image_dirname, sem_seg_dirname in [
        # ("train", "images_detectron2/train", "annotations_detectron2/train"),
        # ("test", "images_detectron2/test", "annotations_detectron2/test"),
        ("train", "images/train2017", "annotations_21/train2017"),
        ("test", "images/val2017", "annotations_21/val2017"),
        # ("train", "images/train2017", "annotations/train2017"),
        # ("test", "images/val2017", "annotations/val2017"),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        name = f"coco_2017_{name}_stuff_164k_sem_seg_scannet_21"
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="jpg")
        )
        MetadataCatalog.get(name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
            stuff_colors=COCO_CATEGORIES_SCANNET_COLORS_21[:],
            **meta,
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_coco_stuff_164k(_root)
register_all_coco_stuff_164k_scannet(_root)
register_all_coco_stuff_164k_scannet_18(_root)
register_all_coco_stuff_164k_scannet_21(_root)
