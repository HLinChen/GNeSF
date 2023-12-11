export CUDA_VISIBLE_DEVICES=1
export DETECTRON2_DATASETS=/data/hlchen/data/
out_dir=/data/hlchen/output/mask2former/scannet_swin_large_in21k_160k_new/

python ./seg2d/train_net.py --dist-url auto \
  --config-file ./seg2d/configs/scannet/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_160k_res640.yaml \
  --resume \
  --num-gpus 1 \
  --eval-only \
  OUTPUT_DIR $out_dir
