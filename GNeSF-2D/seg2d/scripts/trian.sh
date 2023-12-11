export CUDA_VISIBLE_DEVICES=3
export DETECTRON2_DATASETS=/data/hlchen/data/
# export DETECTRON2_DATASETS=/hdd/hlchen/data/

# out_dir=/data/hlchen/output/mask2former/coco_r50_bs16_4gpu # _100scene

# python train_net.py \
#   --config-file configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml \
#   --num-gpus 4 SOLVER.IMS_PER_BATCH 16 SOLVER.BASE_LR 0.00005 \
#   OUTPUT_DIR $out_dir

# #   --num-gpus 8 SOLVER.IMS_PER_BATCH 16 SOLVER.BASE_LR 0.0001 \

out_dir=/data/hlchen/output/mask2former/ade20k_21cls_r50_bs16_4gpu # _100scene

python train_net.py \
  --config-file configs/ade20k/semantic-segmentation/maskformer2_R50_bs16_160k_21cls.yaml \
  --num-gpus 1 SOLVER.IMS_PER_BATCH 16 SOLVER.BASE_LR 0.00005 \
  OUTPUT_DIR $out_dir
