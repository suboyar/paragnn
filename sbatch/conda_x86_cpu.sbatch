#!/usr/bin/env bash
#SBATCH --job-name="conda_x86_cpu"
#SBATCH --nodes=1
#SBATCH --ntasks=1

module purge
module load anaconda3/x86_64/2025.12-2

eval "$(conda shell.bash hook)"
conda env list | grep -qw "pyg_x86_cpu" || conda create -n pyg_x86_cpu python=3.10 -y
eval "$(conda shell.bash activate pyg_x86_cu126)"
pip install torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cpu
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.12.0+cpu.html
pip install ogb

srun python pyg.py --device cpu --epochs 3 --num_layers 3 --hidden_channels 5
