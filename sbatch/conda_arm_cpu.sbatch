#!/usr/bin/env bash
#SBATCH --job-name="conda_arm_cpu"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive

# This needs to be set, or else the DefaultCPUAllocator fails allocating 83kB of data
ulimit -v unlimited

module purge
module load anaconda3/aarch64/2024.10-1

eval "$(conda shell.bash hook)"
conda env list | grep -qw "pyg_arm_cpu" || conda create -n pyg_arm_cpu python=3.10 -y
eval "$(conda shell.bash activate pyg_arm_cpu)"
pip install torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cpu
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.12.0+cpu.html
pip install ogb

python pyg.py --device cpu --epochs 3 --num_layers 3 --hidden_channels 5
