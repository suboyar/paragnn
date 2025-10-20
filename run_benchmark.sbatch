#!/usr/bin/env bash

#SBATCH --job-name="paragnn"
#SBATCH --exclusive
#SBATCH --output=logs/%j_%x.out

set -e

module purge
module load numactl
# TODO: try out gcc-13 with module load gcc/13.2.0

BENCHMARK_TARGET=${BENCHMARK_TARGET:-paragnn}

TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
RESULT_DIR="results/${BENCHMARK_TARGET}/${SLURM_JOB_PARTITION}/${TIMESTAMP}"
BUILD_DIR=build/${SLURM_JOB_PARTITION}

mkdir -p ${RESULT_DIR}

case ${BENCHMARK_TARGET} in
    paragnn)
        EXEC=${BUILD_DIR}/paragnn
        ;;
    *)
        KERNEL_NAME=${BENCHMARK_TARGET}
        EXEC=${BUILD_DIR}/kernel-${KERNEL_NAME}
        ;;
esac

ARCH=$(uname -m)
NOB="./nob_${ARCH}"

if [ ! -f "${NOB}" ]; then
    gcc -o "${NOB}" nob.c
fi

if [[ -n ${KERNEL_NAME} ]]; then
    ${NOB} -kernel ${KERNEL_NAME} -rebuild -release -outdir ${BUILD_DIR}
else
    ${NOB} -rebuild -release -outdir ${BUILD_DIR}
fi

. affinity-functions.sh

case ${SLURM_JOB_PARTITION} in
    defq)
        time defq_affinity ;;
    fpgaq)
        time fpgaq_affinity ;;
    genoaxq)
        time genoaxq_affinity ;;
    gh200q)
        time gh200q_affinity ;;
    milanq)
        time milanq_affinity ;;
    xeonmaxq)
        time xeonmaxq_affinity ;;
    rome16q)
        time rome16q_affinity ;;
    *)
        echo "Unknown partition: ${SLURM_JOB_PARTITION}"
        exit 1
        ;;
esac
