#!/usr/bin/env bash
#SBATCH --job-name="pyg_cpu_benchmark"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive

if [ "$(uname -m)" = "x86_64" ];
then
    module purge
    module load anaconda3/x86_64/2025.12-2
    eval "$(conda shell.bash activate pyg_x86_cpu)"
elif [ "$(uname -m)" = "aarch64" ];
then
    unset LOADEDMODULES
    unset _LMFILES_
    module load anaconda3/aarch64/2024.10-1
    eval "$(conda shell.bash activate pyg_arm_cpu)"
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
    run_cmd python pyg.py --device cpu --benchmark --dataset "$dataset" --num_layers 4 --hidden_channels 256

    echo
    echo "----------------------------------------------------------------"
    echo " ${dataset} (sparse)"
    echo "----------------------------------------------------------------"
    run_cmd python pyg.py --device cpu --benchmark --dataset "$dataset" --num_layers 4 --hidden_channels 256 --sparse
}

run_test "ogbn-arxiv"
run_test "ogbn-products"
