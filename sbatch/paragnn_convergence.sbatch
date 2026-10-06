#!/usr/bin/env bash

#SBATCH --job-name="paragnn_convergence"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive

if [ $(uname -m) = "x86_64" ];
then
    module purge
    module load gcc/cu1281/15.2.0
    module load openblas/dynamic/0.3.30
    module load numactl/gcc/2.0.17
elif [ $(uname -m) = "aarch64" ];
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
run_cmd make -B paragnn BUILDDIR="$BDIR" IMPL=blas SPARSE=cs

BASE_DIR="${RESULTS_DIR:-results}"
PARTITION_DIR="${BASE_DIR}/${SLURM_JOB_PARTITION}"
mkdir -p "$PARTITION_DIR"

echo "----------------------------------------------------------------"
echo " ogbn-arxiv"
echo "----------------------------------------------------------------"
run_cmd ./"$BDIR"/paragnn --dataset ogbn-arxiv --csv --output "${PARTITION_DIR}/ogbn-arxiv.csv"

echo "----------------------------------------------------------------"
echo " ogbn-products"
echo "----------------------------------------------------------------"
run_cmd ./"$BDIR"/paragnn --dataset ogbn-products --csv --output "${PARTITION_DIR}/ogbn-products.csv"

echo "----------------------------------------------------------------"
echo " ogbn-papers100M"
echo "----------------------------------------------------------------"
run_cmd ./"$BDIR"/paragnn --dataset ogbn-papers100M --csv --output "${PARTITION_DIR}/ogbn-papers100M.csv"
