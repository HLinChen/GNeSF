export CUDA_VISIBLE_DEVICES=2

# out_dir=/data/hlchen/output/mask2former/coco_r50_bs16_4gpu
# mkdir -p ${out_dir}/demo/

# cd demo/
# python demo.py --config-file ../configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml \
#   --input /data/hlchen/data/scannet_colmap0/scans/scene0592_00/images/*.png \
#   --output ${out_dir}/demo/ \
#   --opts MODEL.WEIGHTS ${out_dir}/model_final_94dc52.pkl # model_final.pth
# #   [--other-options]


out_dir=/data/hlchen/output/mask2former/scannet_swin_large_in21k_160k_new/
# scene=scene0316_00


# scene=scene0575_02
# scene=scene0599_00
# scene=scene0621_00
# scene=scene0217_00
# scene=scene0139_00
# scene=scene0406_01
# scene=scene0084_00
# scene=scene0647_00
# scene=scene0046_00
# scene=scene0643_00
# scene=scene0414_00 # lab0-0 seg
# scene=scene0050_02 # lab0-1 debug
# scene=scene0300_00 # lab0-2 debug
# scene=scene0606_01 # lab0-3 debug
# scene=scene0329_02 # lab1-2 seg4
# scene=scene0695_02 # lab3-0 seg4
# scene=scene0496_00 # lab3-2 semany1
# scene=scene0144_00 # lab3-2 semany2 ???
# scene=scene0084_01 # lab4-0 seg0
# scene=scene0081_01 # lab4-1 seg3
scene=scene0277_00 # lab4-2 seg4


mkdir -p ${out_dir}/demo/
mkdir -p ${out_dir}/demo/$scene
# img_dir=/data/hlchen/data/hypersim_sim/scenes/ai_054_007_cam_02/color/*.jpg
# img_dir=/data/hlchen/data/scannet_5/scans/scene0616_00/color/*.jpg # scene0050_00 scene0084_00 scene0580_00
img_dir=/data/hlchen/data/scannet/scans/${scene}/color/*.jpg # scene0050_00 scene0084_00 scene0580_00
# img_dir=/data/hlchen/data/Replica_Dataset/office_0/Sequence_1/rgb/*.png # scene0050_00 scene0084_00 scene0580_00
# img_dir=/data/hlchen/data/coco_stuff/images/val2017/*.jpg

cfg_file=../configs/scannet/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_160k_res640.yaml

cd demo/
python demo.py --config-file $cfg_file \
  --input ${img_dir} \
  --output ${out_dir}/demo/${scene} \
  --opts MODEL.WEIGHTS ${out_dir}/model_final.pth # model_0114999.pth # model_final.pth
#   [--other-options]
