#!/usr/bin/env bash

#SBATCH --job-name="paragnn_benchmark_blas_coo"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive

if [ "$(uname -m)" = "x86_64" ];
then
    module purge
    module load gcc/cu1281/15.2.0
    module load openblas/dynamic/0.3.30
    module load numactl/gcc/2.0.17
elif [ "$(uname -m)" = "aarch64" ];
then
    unset LOADEDMODULES
    unset _LMFILES_
    module purge
    module load gcc/cu1302/15.2.0
    module load openblas/dynamic/0.3.30
    module load numactl/gcc/2.0.14
fi

run_cmd() {
    echo "+ $*" >&2
    "$@"
}

BDIR="build_${SLURM_JOB_PARTITION}"

run_cmd ulimit -v unlimited
run_cmd make -B paragnn BUILDDIR="$BDIR" MODE=benchmark IMPL=naive SPARSE=coo > /dev/null

run_test() {
    local dataset="$1"
    echo
    echo "----------------------------------------------------------------"
    echo " ${dataset} (1s)"
    echo "----------------------------------------------------------------"
    run_cmd numactl -C$(./cpumap 1s) ./"$BDIR"/paragnn --dataset "$dataset"

    echo
    echo "----------------------------------------------------------------"
    echo " ${dataset} (1s-smt)"
    echo "----------------------------------------------------------------"
    run_cmd numactl -C$(./cpumap 1s-smt) ./"$BDIR"/paragnn --dataset "$dataset"

    echo
    echo "----------------------------------------------------------------"
    echo " ${dataset} (2s)"
    echo "----------------------------------------------------------------"
    run_cmd numactl -C$(./cpumap 2s) ./"$BDIR"/paragnn --dataset "$dataset"

    echo
    echo "----------------------------------------------------------------"
    echo " ${dataset} (2s-smt)"
    echo "----------------------------------------------------------------"
    run_cmd numactl -C$(./cpumap 2s-smt) ./"$BDIR"/paragnn --dataset "$dataset"

    echo
    echo "----------------------------------------------------------------"
    echo " ${dataset} (scale-1s)"
    echo "----------------------------------------------------------------"
    for c in $(./cpumap scale-1s); do run_cmd numactl -C"$c" ./"$BDIR"/paragnn --dataset "$dataset"; done

    echo
    echo "----------------------------------------------------------------"
    echo " ${dataset} (scale-1s-smt)"
    echo "----------------------------------------------------------------"
    for c in $(./cpumap scale-1s-smt); do run_cmd numactl -C"$c" ./"$BDIR"/paragnn --dataset "$dataset"; done

    echo
    echo "----------------------------------------------------------------"
    echo " ${dataset} (scale-2s)"
    echo "----------------------------------------------------------------"
    for c in $(./cpumap scale-2s); do run_cmd numactl -C"$c" ./"$BDIR"/paragnn --dataset "$dataset"; done

    echo
    echo "----------------------------------------------------------------"
    echo " ${dataset} (scale-2s-smt)"
    echo "----------------------------------------------------------------"
    for c in $(./cpumap scale-2s-smt); do run_cmd numactl -C"$c" ./"$BDIR"/paragnn --dataset "$dataset"; done
}

run_test "ogbn-arxiv"
run_test "ogbn-products"
