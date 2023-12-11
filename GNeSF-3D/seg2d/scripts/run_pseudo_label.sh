export CUDA_VISIBLE_DEVICES=0,1,3

# cfg_file=./configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml
# model_path=/data/hlchen/output/mask2former/coco_r50_bs16_4gpu/model_final_94dc52.pkl
cfg_file=./configs/coco/semantic-segmentation/swin/maskformer2_swin_base_IN21k_384_bs16_50ep_res640.yaml
model_path=/data/hlchen/output/mask2former/coco_25cls_swin_base_in21k/model_0079999.pth
# cfg_file=./configs/hypersim/semantic-segmentation/swin/maskformer2_swin_base_IN21k_384_bs16_160k_res640_30cls.yaml
# model_path=/data/hlchen/output/mask2former/hypersim_31cls_swin_base_in21k/model_final.pth

# python demo/run_pseudo_label.py --config-file ${cfg_file} --opts MODEL.WEIGHTS ${model_path}


python demo/run_pseudo_label.py --config-file ${cfg_file} --i 0 --n 8 --opts MODEL.WEIGHTS ${model_path} &
python demo/run_pseudo_label.py --config-file ${cfg_file} --i 1 --n 8 --opts MODEL.WEIGHTS ${model_path} &
python demo/run_pseudo_label.py --config-file ${cfg_file} --i 2 --n 8 --opts MODEL.WEIGHTS ${model_path} &
python demo/run_pseudo_label.py --config-file ${cfg_file} --i 3 --n 8 --opts MODEL.WEIGHTS ${model_path} &
python demo/run_pseudo_label.py --config-file ${cfg_file} --i 4 --n 8 --opts MODEL.WEIGHTS ${model_path} &
python demo/run_pseudo_label.py --config-file ${cfg_file} --i 5 --n 8 --opts MODEL.WEIGHTS ${model_path} &
python demo/run_pseudo_label.py --config-file ${cfg_file} --i 6 --n 8 --opts MODEL.WEIGHTS ${model_path} &
python demo/run_pseudo_label.py --config-file ${cfg_file} --i 7 --n 8 --opts MODEL.WEIGHTS ${model_path} &
python demo/run_pseudo_label.py --config-file ${cfg_file} --i 8 --n 9 --opts MODEL.WEIGHTS ${model_path} &
# python demo/run_pseudo_label.py --config-file ${cfg_file} --i 9 --n 16 --opts MODEL.WEIGHTS ${model_path} &
# python demo/run_pseudo_label.py --config-file ${cfg_file} --i 10 --n 16 --opts MODEL.WEIGHTS ${model_path} &
# python demo/run_pseudo_label.py --config-file ${cfg_file} --i 11 --n 16 --opts MODEL.WEIGHTS ${model_path} &
# python demo/run_pseudo_label.py --config-file ${cfg_file} --i 12 --n 16 --opts MODEL.WEIGHTS ${model_path} &
# python demo/run_pseudo_label.py --config-file ${cfg_file} --i 13 --n 16 --opts MODEL.WEIGHTS ${model_path} &
# python demo/run_pseudo_label.py --config-file ${cfg_file} --i 14 --n 16 --opts MODEL.WEIGHTS ${model_path} &
# python demo/run_pseudo_label.py --config-file ${cfg_file} --i 15 --n 16 --opts MODEL.WEIGHTS ${model_path}


# python demo/run_pseudo_label.py \
#         --config-file ./configs/ade20k/semantic-segmentation/maskformer2_R50_bs16_160k_21cls.yaml \
#         --opts MODEL.WEIGHTS /hdd/hlchen/output/mask2former/ade20k_21cls_r50_bs16_2gpu_160k_64head_conv_30queries_mask_dim64_fw1024/model_final.pth
