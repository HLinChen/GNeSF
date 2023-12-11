data_path=/hdd/data/scannet/            # change to your own data path
export DETECTRON2_DATASETS=${data_path}
path=/data/hlchen/output/gnesf/3d_official_gt       # change to your own pretrain path

echo $path

model_file=${path}/results_fusion_eval_0

python tools/evaluation_miou.py \
        --model ${model_file} \
        --n_proc 8 \
        --data_path ${data_path} --gt_path ${data_path} \
        --mode val \
        --dataset scannet \
        --n_gpu 1

echo $path
