export CUDA_VISIBLE_DEVICES=0

# out_dir=/data/hlchen/output/mask2former/coco_r50_bs16_4gpu
# mkdir -p ${out_dir}/demo/

# cd demo/
# python demo.py --config-file ../configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml \
#   --input /data/hlchen/data/scannet_colmap0/scans/scene0592_00/images/*.png \
#   --output ${out_dir}/demo/ \
#   --opts MODEL.WEIGHTS ${out_dir}/model_final_94dc52.pkl # model_final.pth
# #   [--other-options]


out_dir=/data/hlchen/output/mask2former/scannet_swin_base_in21k/
mkdir -p ${out_dir}/demo/
# img_dir=/data/hlchen/data/hypersim_sim/scenes/ai_054_007_cam_02/color/*.jpg
img_dir=/data/hlchen/data/scannet_5/scans/scene0616_00/color/*.jpg # scene0050_00 scene0084_00 scene0580_00
# img_dir=/data/hlchen/data/coco_stuff/images/val2017/*.jpg

cfg_file=../configs/scannet/semantic-segmentation/swin/maskformer2_swin_base_IN21k_384_bs16_160k_res640.yaml

cd demo/
python demo.py --config-file $cfg_file \
  --input ${img_dir} \
  --output ${out_dir}/demo/ \
  --opts MODEL.WEIGHTS ${out_dir}/model_final.pth # model_0114999.pth # model_final.pth
#   [--other-options]
