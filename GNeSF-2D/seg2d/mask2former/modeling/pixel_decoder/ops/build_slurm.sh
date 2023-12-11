#!/bin/sh
#SBATCH --job-name=train_atlas
#SBATCH --output=/home/h/hanlin/output/slurm_log/%j.log
#SBATCH --error=/home/h/hanlin/output/slurm_log/%j.err

#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=64000 # 256GB
#SBATCH --partition=medium
#SBATCH --nodelist=xgpc7
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=8

echo "$state Start"
echo Time is `date`
echo "Directory is ${PWD}"
echo "This job runs on the following nodes: ${SLURM_JOB_NODELIST}"

nvidia-smi

# nvcc -V

# ls /usr/local/

# conda install numpy -y

# pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

# pip install cython scipy shapely timm h5py submitit

# pip install scikit-image

# pip install opencv-python

# python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# pip install pytorch-lightning

export CUDA_HOME=/usr/local/cuda # /usr/local/cuda-10.2

#  export PATH="/usr/local/cuda-10.0/bin:$PATH"
#  export LD_LIBRARY_PATH="/usr/local/cuda-10.0/lib64:$LD_LIBRARY_PATH"

sh make.sh
# sh ./scripts/train_seg2d_slurm.sh # scannet/scene0005_00 partition0 0

# python try.py
