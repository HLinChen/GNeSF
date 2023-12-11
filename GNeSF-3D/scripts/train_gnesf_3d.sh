#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
export DETECTRON2_DATASETS=/data/hlchen/data/       # change to your own data path

python -m torch.distributed.launch --nproc_per_node=1 --master_port=23450 main.py --cfg ./config/train_gnesf_3d.yaml
