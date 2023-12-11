
export CUDA_VISIBLE_DEVICES=3

# psnr

python eval/eval_psnr_gnesf.py --config ./configs/scannet_gnesf_2d.txt \
                    --expname 2d --mode val # train_eval

                    # --mode val
