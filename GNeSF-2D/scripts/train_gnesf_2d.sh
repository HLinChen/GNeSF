export CUDA_VISIBLE_DEVICES=1

python -m torch.distributed.launch --nproc_per_node=1 --master_port 29000 train.py \
                            --expname 2d \
                            --i_print 1000 --distributed \
                            --config configs/scannet_gnesf_2d.txt \
                            --num_source_views 8
                            # --expname scannet_sem_m2former_large_scannet_20cls_1gpu_prob_nview8 \
