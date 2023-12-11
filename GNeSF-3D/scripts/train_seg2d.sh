# export CUDA_VISIBLE_DEVICES=2,3
export DETECTRON2_DATASETS=/data/hlchen/data/

out_dir=/data/hlchen/output/gnesf/scannet_swin_base_in21k/

python ./seg2d/train_net.py --dist-url auto \
  --config-file ./seg2d/configs/scannet/semantic-segmentation/swin/maskformer2_swin_base_IN21k_384_bs16_160k_res640.yaml \
  --resume \
  --num-gpus 4 \
  OUTPUT_DIR $out_dir