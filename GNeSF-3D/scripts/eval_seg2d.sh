export DETECTRON2_DATASETS=/ssd/hlchen/

out_dir=/data/hlchen/output/gnesf/scannet_swin_base_in21k/ # _fw1024 # _30queries # _100scene

python ./seg2d/train_net.py --dist-url auto \
  --config-file ./seg2d/configs/scannet/semantic-segmentation/swin/maskformer2_swin_base_IN21k_384_bs16_160k_res640.yaml \
  --resume \
  --num-gpus 4 \
  --eval-only \
  OUTPUT_DIR $out_dir \
  TEST.AUG.ENABLED True
