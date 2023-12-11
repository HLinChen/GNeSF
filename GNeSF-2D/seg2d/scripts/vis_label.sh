# export DETECTRON2_DATASETS=/data/hlchen/data/
export DETECTRON2_DATASETS=/hdd/hlchen/data/

# data_set=scannet # scannet
data_set=coco # scannet
out_dir=/data/hlchen/output/vis_label_${data_set}/

# cfg_file=../configs/$data_set/semantic-segmentation/maskformer2_R50_bs16_160k.yaml
cfg_file=../configs/$data_set/semantic-segmentation/maskformer2_R50_bs16_160k_18cls.yaml
# cfg_file=../configs/$data_set/semantic-segmentation/maskformer2_R50_bs16_50ep.yaml
# cfg_file=../configs/$data_set/semantic-segmentation/maskformer2_R50_bs16_50ep.yaml
# cfg_file=../configs/$data_set/semantic-segmentation/maskformer2_R50_bs16_160k_21cls.yaml

cd demo/
python vis_label.py --config-file $cfg_file \
  --output ${out_dir} \
  --opts SOLVER.IMS_PER_BATCH 1
