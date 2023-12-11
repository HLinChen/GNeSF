# export DETECTRON2_DATASETS=/data/hlchen/data/scannet
export DETECTRON2_DATASETS=/data/hlchen/data/
export CUDA_VISIBLE_DEVICES=3

expname=2d

python eval/eval_miou.py --expname $expname \
        --config configs/scannet_gnesf_2d.txt \
        --mode train_eval \
        --num_source_views 12 \
        --show
        