# GNeSF: Generalizable Neural Semantic Fields
<!-- ### [Project Page](https://zju3dv.github.io/neuralrecon) | [Paper](https://arxiv.org/pdf/2104.00681.pdf) -->
### [Paper](https://arxiv.org/pdf/2310.15712.pdf)
<br/>

> GNeSF: Generalizable Neural Semantic Fields
> [Hanlin Chen](https://hlinchen.github.io/hlchen/), [Chen Li](https://chaneyddtt.github.io/), [Mengqi Guo](https://scholar.google.com/citations?user=Qa4BlOoAAAAJ&hl=en), [Zhiwen Yan](https://jokeryan.github.io/about/), [Zhiwen Yan](https://www.comp.nus.edu.sg/~leegh/)  
> NeurIPS 2023

## How to Use

### Installation
```shell
# Ubuntu 20.04 and above is recommended.
sudo apt install libsparsehash-dev  # you can try to install sparsehash with conda if you don't have sudo privileges.
conda env create -f environment.yaml
conda activate gnesf
```

### Data Preperation for ScanNet
Download and extract ScanNet by following the instructions provided at http://www.scan-net.org/.
<details>
  <summary>[Expected directory structure of ScanNet (click to expand)]</summary>

You can obtain the train/val/test split information from [here](https://github.com/ScanNet/ScanNet/tree/master/Tasks/Benchmark).
```
DATAROOT
└───scannet
│   └───scans
│   |   └───scene0000_00
│   |       └───color
│   |       │   │   0.jpg
│   |       │   │   1.jpg
│   |       │   │   ...
│   |       │   ...
│   └───scans_test
│   |   └───scene0707_00
│   |       └───color
│   |       │   │   0.jpg
│   |       │   │   1.jpg
│   |       │   │   ...
│   |       │   ...
|   └───scannetv2_test.txt
|   └───scannetv2_train.txt
|   └───scannetv2_val.txt
```
</details>

Next run the data preparation script which parses the raw data format into the processed pickle format.
This script also generates the ground truth TSDFs using TSDF Fusion.  
<details>
  <summary>[Data preparation script]</summary>

We also compress all camera information into 'cam_info_all.pth'. You can download it from the [link](https://drive.google.com/file/d/1ia737h8ELF5OMEjiguN6ty_nHfuueIq_/view?usp=drive_link). And then put it under the path to scannet.

```bash
cd GNeSF-3D
# Change PATH_TO_SCANNET and OUTPUT_PATH accordingly.
# For the training/val split:
python tools/tsdf_fusion/generate_gt.py --data_path PATH_TO_SCANNET --save_name all_tsdf_9 --window_size 9
# For the test split
python tools/tsdf_fusion/generate_gt.py --test --data_path PATH_TO_SCANNET --save_name all_tsdf_9 --window_size 9
```
</details>


### Training and Evaluation

Our method relies on pretrained Mask2Former to performa semantic view synthesis or 3D Semantic Segmentation. Please download the pretrained model from the [link](https://drive.google.com/file/d/1CGwrc31S4aI8_01JiDWi1nX9t2AzShtC/view?usp=sharing) first. And then change the path in config file. For different tasks, please use different codes. For Semantic View Synthesis, please read README.md in GNeSF-2D. For 3D Semantic Segmentation, please read README.md in GNeSF-3D.


## <a name="CitingMask2Former"></a>Citing GNeSF

If you use GNeSF in your research, please use the following BibTeX entry.

```BibTeX
@article{chen2023gnesf,
  title={GNeSF: Generalizable Neural Semantic Fields},
  author={Chen, Hanlin and Li, Chen and Guo, Mengqi and Yan, Zhiwen and Lee, Gim Hee},
  journal={arXiv preprint arXiv:2310.15712},
  year={2023}
}
```


## Acknowledgement

Code is largely based on NeuralRecon (https://zju3dv.github.io/neuralrecon/), IBRNet (https://ibrnet.github.io/), and MaskFormer (https://github.com/facebookresearch/MaskFormer).