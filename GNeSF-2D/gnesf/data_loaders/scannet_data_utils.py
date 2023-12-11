import os
import csv
import numpy as np


def load_scannet_nyu40_mapping(path):
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


def load_scannet_label_mapping(path):
    """ Returns a dict mapping scannet category label strings to scannet Ids

    scene****_**.aggregation.json contains the category labels as strings 
    so this maps the strings to the integer scannet Id

    Args:
        path: Path to the original scannet data.
            This is used to get scannetv2-labels.combined.tsv

    Returns:
        mapping: A dict from strings to ints
            example:
                {'wall': 1,
                'chair: 2,
                'books': 22}

    """

    mapping = {}
    with open(os.path.join(path, 'scannetv2-labels.combined.tsv')) as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter='\t')
        for i, line in enumerate(tsvreader):
            if i==0:
                continue
            scannet_id, name = int(line[0]), line[1]
            mapping[name] = scannet_id

    return mapping


# color palette for nyu40 labels
nyu40_colour_code = np.array([
       (0, 0, 0),

       (174, 199, 232),		# wall
       (152, 223, 138),		# floor
       (31, 119, 180), 		# cabinet
       (255, 187, 120),		# bed
       (188, 189, 34), 		# chair

       (140, 86, 75),  		# sofa
       (255, 152, 150),		# table
       (214, 39, 40),  		# door
       (197, 176, 213),		# window
       (148, 103, 189),		# bookshelf

       (196, 156, 148),		# picture
       (23, 190, 207), 		# counter
       (178, 76, 76),       # blinds
       (247, 182, 210),		# desk
       (66, 188, 102),      # shelves

       (219, 219, 141),		# curtain
       (140, 57, 197),    # dresser
       (202, 185, 52),      # pillow
       (51, 176, 203),    # mirror
       (200, 54, 131),      # floor

       (92, 193, 61),       # clothes
       (78, 71, 183),       # ceiling
       (172, 114, 82),      # books
       (255, 127, 14), 		# refrigerator
       (91, 163, 138),      # tv

       (153, 98, 156),      # paper
       (140, 153, 101),     # towel
       (158, 218, 229),		# shower curtain
       (100, 125, 154),     # box
       (178, 127, 135),       # white board

       (120, 185, 128),       # person
       (146, 111, 194),     # night stand
       (44, 160, 44),  		# toilet
       (112, 128, 144),		# sink
       (96, 207, 209),      # lamp

       (227, 119, 194),		# bathtub
       (213, 92, 176),      # bag
       (94, 106, 211),      # other struct
       (82, 84, 163),  		# otherfurn
       (100, 85, 144)       # other prop
    ]).astype(np.uint8)

# color palette for scannet20 labels
scannet20_colour_code = np.array(
    [
        (0, 0, 0),
        (174, 199, 232),  # wall
        (152, 223, 138),  # floor
        (31, 119, 180),  # cabinet
        (255, 187, 120),  # bed
        (188, 189, 34),  # chair
        (140, 86, 75),  # sofa
        (255, 152, 150),  # table
        (214, 39, 40),  # door
        (197, 176, 213),  # window
        (148, 103, 189),  # bookshelf
        (196, 156, 148),  # picture
        (23, 190, 207),  # counter
        (247, 182, 210),  # desk
        (219, 219, 141),  # curtain
        (255, 127, 14),  # refrigerator
        (158, 218, 229),  # shower curtain
        (44, 160, 44),  # toilet
        (112, 128, 144),  # sink
        (227, 119, 194),  # bathtub
        (82, 84, 163),  # otherfurn
    ]).astype(np.uint8)

# color palette for nyu40 labels
scannet21_colour_code = np.array([
       (0, 0, 0),

       (174, 199, 232),		# wall
       (152, 223, 138),		# floor
       (31, 119, 180), 		# cabinet
       (255, 187, 120),		# bed
       (188, 189, 34), 		# chair

       (140, 86, 75),  		# sofa
       (255, 152, 150),		# table
       (214, 39, 40),  		# door
       (197, 176, 213),		# window
       (23, 190, 207), 		# counter
       (66, 188, 102),      # shelves

       (219, 219, 141),		# curtain
       (78, 71, 183),       # ceiling
       (255, 127, 14), 		# refrigerator
       (91, 163, 138),      # tv
       (120, 185, 128),       # person
       
       (44, 160, 44),  		# toilet
       (112, 128, 144),		# sink
       (96, 207, 209),      # lamp
       (213, 92, 176),      # bag
       (100, 85, 144)       # other prop
    
    ]).astype(np.uint8)

