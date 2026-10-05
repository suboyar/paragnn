#!/usr/bin/env bash
#SBATCH --job-name="conda_x86_gpu"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

module purge
module load cuda12.6/blas/12.6.3
module load cuda12.6/fft/12.6.3
module load cuda12.6/nsight/12.6.3
module load cuda12.6/profiler/12.6.3
module load cuda12.6/toolkit/12.6.3
module load anaconda3/x86_64/2025.12-2

eval "$(conda shell.bash hook)"
conda env list | grep -qw "pyg_x86_cu126" || conda create -n pyg_x86_cu126 python=3.10 -y
eval "$(conda shell.bash activate pyg_x86_cu126)"
pip install torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cu126
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.12.0+cu126.html
pip install ogb

python pyg.py --device gpu --epochs 3 --num_layers 3 --hidden_channels 5
