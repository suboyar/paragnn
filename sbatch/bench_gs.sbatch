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
CFLAGS="-DEXPORT_CSV"
if cores=$(./cpumap 1s); then
    { set -x; } 2>/dev/null
    make -B bench-gs CFLAGS="$CFLAGS" BUILDDIR="$BDIR" > /dev/null
    numactl -C"$cores" ./${BDIR}/bench-gs -ntimes 5000
    { set +x; } 2>/dev/null
fi

if cores=$(./cpumap 1s-smt); then
    { set -x; } 2>/dev/null
    make -B bench-gs CFLAGS="$CFLAGS" BUILDDIR="$BDIR" > /dev/null
    numactl -C"$cores" ./${BDIR}/bench-gs -ntimes 5000
    { set +x; } 2>/dev/null
fi

if cores=$(./cpumap 2s); then
    { set -x; } 2>/dev/null
    make -B bench-gs CFLAGS="$CFLAGS" BUILDDIR="$BDIR" > /dev/null
    numactl -C"$cores" ./${BDIR}/bench-gs -ntimes 5000
    { set +x; } 2>/dev/null
fi

if cores=$(./cpumap 2s-smt); then
    { set -x; } 2>/dev/null
    make -B bench-gs CFLAGS="$CFLAGS" BUILDDIR="$BDIR" > /dev/null
    numactl -C"$cores" ./${BDIR}/bench-gs -ntimes 5000
    { set +x; } 2>/dev/null
fi
