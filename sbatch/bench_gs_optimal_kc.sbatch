#!/usr/bin/env bash

#SBATCH --job-name="bench_gs_optimal_kc"
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

BDIR="build_${SLURM_JOB_PARTITION}"

cores=$(./cpumap max)
echo "----------------------------------------------------------------"
echo " Testing for fma latency"
echo "----------------------------------------------------------------"
for KC in 1 2 4 8 16 32 64 128 256 512 1024
do
    { set -x; } 2>/dev/null
    make -B bench-gs CFLAGS="-DFMA_LAT -DKC=${KC}" BUILDDIR="$BDIR" > /dev/null
    numactl -C"$cores" ./${BDIR}/benchmark/bench-gs -ntimes 5000 -dims 256
    { set +x; } 2>/dev/null
done

echo "----------------------------------------------------------------"
echo " Testing for computation-to-load"
echo "----------------------------------------------------------------"
for KC in 1 2 4 8 16 32 64 128 256 512 1024
do
    { set -x; } 2>/dev/null
    make -B bench-gs CFLAGS="-DKC=${KC}" BUILDDIR="$BDIR" > /dev/null
    numactl -C"$cores" ./${BDIR}/benchmark/bench-gs -ntimes 5000 -dims 256
    { set +x; } 2>/dev/null
done
