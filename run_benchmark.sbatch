#!/usr/bin/env bash

# Slurm benchmark runner: compiles and runs a target on a single socket.
# Submitted via: ./nob -slurm [-p PARTITION] [-target TARGET] [-D MACRO]

set -euo pipefail

module purge
module load numactl
module load openblas/dynamic/0.3.30

set -x

TARGET=${BENCHMARK_TARGET:-paragnn}
COMPDIR=build/$SLURM_JOB_PARTITION/$TARGET
EXEC=$COMPDIR/${TARGET}

NOB="./nob-$(uname -m)"
[[ -f $NOB ]] || gcc -ggdb -o "$NOB" nob.c

echo "Compiling $TARGET (${BENCHMARK_CONFIG:-default})"
$NOB -release -target $TARGET -o $COMPDIR $BENCHMARK_MACROS

# 1 socket without smt
case ${SLURM_JOB_PARTITION} in
    defq)     CPUSET=0-31 ;;
    fpgaq)    CPUSET=0-15 ;;
    genoaxq)  CPUSET=0-63 ;;
    gh200q)   CPUSET=0-71 ;;
    habanaq)  CPUSET=0-35 ;;
    milanq)   CPUSET=0-63 ;;
    rome16q)  CPUSET=0-15 ;;
    xeonmaxq) CPUSET=0-55 ;;
    huaq)     CPUSET=0-63 ;;
    armq)     CPUSET=0-31 ;;
    *)
        echo "Unknown partition: ${SLURM_JOB_PARTITION}" >&2
        exit 1
        ;;
esac

case $TARGET in
    paragnn)
        numactl -C$CPUSET $EXEC -csv stdout -epochs 1000 -layers 4 -channels 256 -lr 0.01
        ;;
    tsmm_tn)
        numactl -C$CPUSET $EXEC -csv stdout
        ;;
    *)
        echo "Unknown target: ${TARGET}" >&2
        exit 1
        ;;
esac
