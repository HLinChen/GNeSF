export CUDA_VISIBLE_DEVICES=0
PATH_TO_SCANNET=/hdd/hlchen/scannet/

# Change PATH_TO_SCANNET and OUTPUT_PATH accordingly.
# For the training/val split:
python tools/tsdf_fusion/generate_gt.py --data_path ${PATH_TO_SCANNET} --save_name all_tsdf_9 --window_size 9


# For the test split
python tools/tsdf_fusion/generate_gt.py --test --data_path ${PATH_TO_SCANNET} --save_name all_tsdf_9 --window_size 9
