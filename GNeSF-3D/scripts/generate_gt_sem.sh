# export CUDA_VISIBLE_DEVICES=1,2,3
# # PATH_TO_SCANNET=/hdd/hlchen/scannet/
PATH_TO_SCANNET=/data/hlchen/data/scannet/

# Change PATH_TO_SCANNET and OUTPUT_PATH accordingly.
# For the training/val split:
# python tools/tsdf_fusion/generate_gt.py --data_path ${PATH_TO_SCANNET} --save_name all_tsdf_4 --window_size 4
python tools/tsdf_fusion/generate_gt.py \
        --data_path ${PATH_TO_SCANNET} --save_name all_tsdf_9_21cls \
        --window_size 9 --enable_sem --num_cls 21 --n_gpu 3 --n_proc 3 --save_mesh



For the test split
python tools/tsdf_fusion/generate_gt.py --test --data_path ${PATH_TO_SCANNET} --save_name all_tsdf_9 --window_size 9
