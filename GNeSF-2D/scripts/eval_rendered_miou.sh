export CUDA_VISIBLE_DEVICES=0
export DETECTRON2_DATASETS=/data/hlchen/data # /scannet

python eval/eval_rendered_miou.py
