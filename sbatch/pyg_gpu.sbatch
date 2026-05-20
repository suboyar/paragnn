#!/usr/bin/env bash

#SBATCH --job-name="pyg_gpu"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

if [[ $(uname -m) = "aarch64" ]]
then
    unset LOADEDMODULES
    unset _LMFILES_
fi

module purge
module load slurm/21.08.8

skip_papers100M=0
if [[ $(uname -m) = "x86_64" ]]
then
    module load anaconda3/x86_64/2024.10-1
    if [[ $(uname -n) = "g001" ]] # dgx2q
    then
        module load cuda12.2/toolkit/12.2.2
        module load cuda12.2/blas/12.2.2
        module load cuda12.2/nsight/12.2.2
        module load cuda12.2/profiler/12.2.2
        module load cuda12.2/fft/12.2.20.1
        eval "$(conda shell.bash activate py12_torch2.5.1_cu121_pyg)"
        skip_papers100M=1
    elif [[ $(uname -n) = "g002" ]] # hgx2q
    then
        module load cuda13.0/blas/13.0.2
        module load cuda13.0/fft/13.0.2
        module load cuda13.0/nsight/13.0.2
        module load cuda13.0/nvml/13.0.2/580.95.05
        module load cuda13.0/profiler/13.0.2
        module load cuda13.0/toolkit/13.0.2
        eval "$(conda shell.bash activate py12_torch2.11.0_cu130_pyg)"
    elif [[ $(uname -n) = "n013" || $(uname -n) = "n014" ]] # a100q
    then
        module load cuda13.0/blas/13.0.2
        module load cuda13.0/fft/13.0.2
        module load cuda13.0/nsight/13.0.2
        module load cuda13.0/nvml/13.0.2/580.95.05
        module load cuda13.0/profiler/13.0.2
        module load cuda13.0/toolkit/13.0.2
        eval "$(conda shell.bash activate py12_torch2.11.0_cu130_pyg)"
        skip_papers100M=1
    fi
elif [[ $(uname -m) = "aarch64" ]]
then
    module load anaconda3/aarch64/2024.10-1
    if [[ $(uname -n) = "n009" || $(uname -n) = "n010" || $(uname -n) = "n011" || $(uname -n) = "n012" ]] # a40q
    then
        module load cuda13.0/blas/13.0.2
        module load cuda13.0/fft/13.0.2
        module load cuda13.0/nsight/13.0.2
        module load cuda13.0/nvml/13.0.2/580.95.05
        module load cuda13.0/profiler/13.0.2
        module load cuda13.0/toolkit/13.0.2
        eval "$(conda shell.bash activate py12_torch2.11.0_cu130_pyg_aarch64)"
        skip_papers100M=1
    fi
    if [[ $(uname -n) = "gh001" || $(uname -n) = "gh002" ]] # gh200q
    then
        module load cuda13.0/blas/13.0.2
        module load cuda13.0/fft/13.0.2
        module load cuda13.0/nsight/13.0.2
        module load cuda13.0/nvml/13.0.2/580.95.05
        module load cuda13.0/profiler/13.0.2
        module load cuda13.0/toolkit/13.0.2
        eval "$(conda shell.bash activate py12_torch2.11.0_cu130_pyg_aarch64)"
    fi
fi

echo "----------------------------------------------------------------------"
python pyg.py --device gpu --dataset arxiv --datadir /home/sboyar/D1/pyg-dataset
echo -e "----------------------------------------------------------------------\n"

echo "----------------------------------------------------------------------"
python pyg.py --device gpu --dataset products --datadir /home/sboyar/D1/pyg-dataset
echo -e "----------------------------------------------------------------------\n"

if [[ $skip_papers100M = 0 ]]
then
    echo "----------------------------------------------------------------------"
    python pyg.py --device gpu --dataset papers100M --datadir /home/sboyar/D1/pyg-dataset
    echo -e "----------------------------------------------------------------------\n"
fi
