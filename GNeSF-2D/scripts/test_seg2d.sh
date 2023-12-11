export DETECTRON2_DATASETS=/data/hlchen/data/
out_dir=/data/hlchen/output/mask2former/scannet_swin_base_in21k_21cls_10/
# export CUDA_VISIBLE_DEVICES=2,3


python ./seg2d/train_net.py \
        --config-file ./seg2d/configs/scannet/semantic-segmentation/swin/maskformer2_swin_base_IN21k_384_bs16_160k_res640_21cls.yaml \
        --num-gpus 1 \
        --eval-only \
        --dist-url auto \
        --resume \
        OUTPUT_DIR $out_dir
        # SOLVER.IMS_PER_BATCH 16 SOLVER.BASE_LR 0.01 SOLVER.MAX_ITER 360000 \

