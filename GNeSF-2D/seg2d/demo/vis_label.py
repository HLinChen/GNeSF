import multiprocessing as mp
from collections import deque
import argparse
import glob
import multiprocessing as mp
import os
import time

import cv2
import torch
import matplotlib.colors as mplc
import numpy as np
import tqdm
import sys
from PIL import Image
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from mask2former import add_maskformer2_config
from train_net import Trainer
from predictor import MyVisualizer, AsyncPredictor

# constants
WINDOW_NAME = "mask2former demo"


class VisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.parallel = parallel

    def run_on_image(self, image, label):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.
        Returns:
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        # image = image[:, :, ::-1]
        visualizer = MyVisualizer(image, self.metadata, instance_mode=self.instance_mode)
        if "panoptic_seg" in label:
            panoptic_seg, segments_info = label["panoptic_seg"]
            vis_output = visualizer.draw_panoptic_seg_predictions(
                panoptic_seg.to(self.cpu_device), segments_info
            )
        else:
            if "sem_seg" in label:
                vis_output = visualizer.draw_sem_seg(
                    label["sem_seg"].to(self.cpu_device)
                )
            if "instances" in label:
                instances = label["instances"].to(self.cpu_device)
                vis_output = visualizer.draw_instance_predictions(predictions=instances)

        return vis_output

    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break

    def run_on_video(self, video):
        """
        Visualizes predictions on frames of the input video.
        Args:
            video (cv2.VideoCapture): a :class:`VideoCapture` object, whose source can be
                either a webcam or a video file.
        Yields:
            ndarray: BGR visualizations of each video frame.
        """
        video_visualizer = VideoVisualizer(self.metadata, self.instance_mode)

        def process_predictions(frame, predictions):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if "panoptic_seg" in predictions:
                panoptic_seg, segments_info = predictions["panoptic_seg"]
                vis_frame = video_visualizer.draw_panoptic_seg_predictions(
                    frame, panoptic_seg.to(self.cpu_device), segments_info
                )
            elif "instances" in predictions:
                predictions = predictions["instances"].to(self.cpu_device)
                vis_frame = video_visualizer.draw_instance_predictions(frame, predictions)
            elif "sem_seg" in predictions:
                vis_frame = video_visualizer.draw_sem_seg(
                    frame, predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )

            # Converts Matplotlib RGB format to OpenCV BGR format
            vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
            return vis_frame

        frame_gen = self._frame_from_video(video)
        if self.parallel:
            buffer_size = self.predictor.default_buffer_size

            frame_data = deque()

            for cnt, frame in enumerate(frame_gen):
                frame_data.append(frame)
                self.predictor.put(frame)

                if cnt >= buffer_size:
                    frame = frame_data.popleft()
                    predictions = self.predictor.get()
                    yield process_predictions(frame, predictions)

            while len(frame_data):
                frame = frame_data.popleft()
                predictions = self.predictor.get()
                yield process_predictions(frame, predictions)
        else:
            for frame in frame_gen:
                yield process_predictions(frame, self.predictor(frame))


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="maskformer2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    os.makedirs(args.output, exist_ok=True)
    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)
    
    dataloader = Trainer.build_test_loader(cfg, cfg.DATASETS.TEST[0])

        # args.input = glob.glob(os.path.expanduser(args.input[0]))
        # assert args.input, "The input path(s) was not found"
    # for path in tqdm.tqdm(args.input, disable=not args.output):
    #     # use PIL, to be consistent with evaluation
    #     img = read_image(path, format="BGR")
    for i, data in enumerate(tqdm.tqdm(dataloader)):
        data = data[0]
        label = {'sem_seg': data['sem_seg']} # , 'panoptic_seg': data['panoptic_seg'], 'instances': data['instances'
        img = data['image'].permute(1, 2, 0).numpy()
        path = data['file_name']
        # if i == 21:
        #     print(path)
        #     exit()
        # print(img.shape)
        # exit()
        # sem_seg = 
        start_time = time.time()
        visualized_output = demo.run_on_image(img, label)

        if args.output:
            # if os.path.isdir(args.output):
            #     assert os.path.isdir(args.output), args.output
            #     out_filename = os.path.join(args.output, os.path.basename(path))
            # else:
            #     assert len(args.input) == 1, "Please specify a directory with args.output"
            # out_filename = os.path.join(args.output, os.path.basename(path))
            # out_filename = os.path.join(args.output, str(i))
            out_filename = os.path.join(args.output, '_'.join(path[:-4].split('/')[6:]))
            visualized_output.save(out_filename)
            # print(label["sem_seg"].numpy().astype(np.uint8))
            # Image.fromarray(label["sem_seg"].numpy().astype(np.uint8)).save(out_filename+'.png')
        else:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
            if cv2.waitKey(0) == 27:
                break  # esc to quit

