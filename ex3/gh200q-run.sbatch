#!/usr/bin/env bash

#SBATCH --job-name="paragnn_benchmark"
#SBATCH --exclusive
#SBATCH --partition=gh200q
#SBATCH --output=gh200q-%j-%x-stdout.txt
#SBATCH --error=gh200q-%j-%x-stderr.txt

set -e

# EXPERIMENT is passed via --export from nob.c
EXPERIMENT="${EXPERIMENT:-baseline}"

TIMESTAMP=$(date +%Y%m%dT%H%M%S)
RESULT_DIR="results/${EXPERIMENT}/${SLURM_JOB_PARTITION}/${TIMESTAMP}"
EXEC=./build/${EXPERIMENT}/${SLURM_JOB_PARTITION}/paragnn

module load numactl

numactl -C0 ${EXEC} -output ${RESULT_DIR}/1cores.csv
numactl -C0-1 ${EXEC} -output ${RESULT_DIR}/2cores.csv
numactl -C0-3 ${EXEC} -output ${RESULT_DIR}/4cores.csv
numactl -C0-7 ${EXEC} -output ${RESULT_DIR}/8cores.csv
numactl -C0-15 ${EXEC} -output ${RESULT_DIR}/16cores.csv
numactl -C0-31 ${EXEC} -output ${RESULT_DIR}/32cores.csv
numactl -C0-63 ${EXEC} -output ${RESULT_DIR}/64cores.csv
# numactl -C0-71 ${EXEC} -output ${RESULT_DIR}/064cores.csv
