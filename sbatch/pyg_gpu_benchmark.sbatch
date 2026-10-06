#!/usr/bin/env bash
#SBATCH --job-name="pyg_gpu_benchmark"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive

if [ "$(uname -m)" = "x86_64" ];
then
    module purge
    module load cuda12.6/blas/12.6.3
    module load cuda12.6/fft/12.6.3
    module load cuda12.6/nsight/12.6.3
    module load cuda12.6/profiler/12.6.3
    module load cuda12.6/toolkit/12.6.3
    module load anaconda3/x86_64/2025.12-2
    eval "$(conda shell.bash activate pyg_x86_cu126)"
elif [ "$(uname -m)" = "aarch64" ];
then
    unset LOADEDMODULES
    unset _LMFILES_
    module load gcc/cu1232/13.3.0
    module load cuda12.6/blas/12.6.3
    module load cuda12.6/fft/12.6.3
    module load cuda12.6/nsight/12.6.3
    module load cuda12.6/profiler/12.6.3
    module load cuda12.6/toolkit/12.6.3
    module load anaconda3/aarch64/2024.10-1
    eval "$(conda shell.bash activate pyg_arm_cu126)"
fi

run_cmd() {
    echo "+ $*" >&2
    "$@"
}

run_cmd ulimit -v unlimited

run_test() {
    local dataset="$1"
    echo
    echo "----------------------------------------------------------------"
    echo " ${dataset} (not sparse)"
    echo "----------------------------------------------------------------"
    run_cmd python pyg.py --device gpu --benchmark --dataset "$dataset" --num_layers 4 --hidden_channels 256

    echo
    echo "----------------------------------------------------------------"
    echo " ${dataset} (sparse)"
    echo "----------------------------------------------------------------"
    run_cmd python pyg.py --device gpu --benchmark --dataset "$dataset" --num_layers 4 --hidden_channels 256 --sparse
}

run_test "ogbn-arxiv"
run_test "ogbn-products"
run_test "ogbn-papers100M"
