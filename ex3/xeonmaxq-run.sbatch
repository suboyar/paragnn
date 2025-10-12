#!/usr/bin/env bash

#SBATCH --job-name="paragnn_benchmark"
#SBATCH --exclusive
#SBATCH --partition=xeonmaxq
#SBATCH --output=xeonmaxq-%j-%x-stdout.txt
#SBATCH --error=xeonmaxq-%j-%x-stderr.txt

set -e

# EXPERIMENT is passed via --export from nob.c
EXPERIMENT="${EXPERIMENT:-baseline}"

TIMESTAMP=$(date +%Y%m%dT%H%M%S)
RESULT_DIR="results/${EXPERIMENT}/${SLURM_JOB_PARTITION}/${TIMESTAMP}"
EXEC=./build/${EXPERIMENT}/${SLURM_JOB_PARTITION}/paragnn

module load numactl

# Single socket
numactl -C0 ${EXEC} -output ${RESULT_DIR}/01c_1s.csv
numactl -C0-1 ${EXEC} -output ${RESULT_DIR}/02c_1s.csv
numactl -C0-3 ${EXEC} -output ${RESULT_DIR}/04c_1s.csv
numactl -C0-7 ${EXEC} -output ${RESULT_DIR}/08c_1s.csv
numactl -C0-15 ${EXEC} -output ${RESULT_DIR}/16c_1s.csv
numactl -C0-31 ${EXEC} -output ${RESULT_DIR}/32c_1s.csv
# numactl -C0-55 ${EXEC} -output ${RESULT_DIR}/56c_1s.csv

# Dual socket
numactl -C0,56 ${EXEC} -output ${RESULT_DIR}/02c_2s.csv
numactl -C0-1,56-57 ${EXEC} -output ${RESULT_DIR}/04c_2s.csv
numactl -C0-3,56-59 ${EXEC} -output ${RESULT_DIR}/08c_2s.csv
numactl -C0-7,56-63 ${EXEC} -output ${RESULT_DIR}/16c_2s.csv
numactl -C0-15,56-71 ${EXEC} -output ${RESULT_DIR}/32c_2s.csv
numactl -C0-31,56-87 ${EXEC} -output ${RESULT_DIR}/64c_2s.csv
# numactl -C0-55,56-111 ${EXEC} -output ${RESULT_DIR}/112c_2s.csv

# Hyperthreading - Single socket
numactl -C0,112 ${EXEC} -output ${RESULT_DIR}/01c_1s_smt.csv
numactl -C0-1,112-113 ${EXEC} -output ${RESULT_DIR}/02c_1s_smt.csv
numactl -C0-3,112-115 ${EXEC} -output ${RESULT_DIR}/04c_1s_smt.csv
numactl -C0-7,112-119 ${EXEC} -output ${RESULT_DIR}/08c_1s_smt.csv
numactl -C0-15,112-127 ${EXEC} -output ${RESULT_DIR}/16c_1s_smt.csv
numactl -C0-31,112-143 ${EXEC} -output ${RESULT_DIR}/32c_1s_smt.csv
# numactl -C0-55,112-167 ${EXEC} -output ${RESULT_DIR}/56c_1s_smt.csv

# Hyperthreading - Dual socket
numactl -C0,56,112,168 ${EXEC} -output ${RESULT_DIR}/02c_2s_smt.csv
numactl -C0-1,56-57,112-113,168-169 ${EXEC} -output ${RESULT_DIR}/04c_2s_smt.csv
numactl -C0-3,56-59,112-115,168-171 ${EXEC} -output ${RESULT_DIR}/08c_2s_smt.csv
numactl -C0-7,56-63,112-119,168-175 ${EXEC} -output ${RESULT_DIR}/16c_2s_smt.csv
numactl -C0-15,56-71,112-127,168-183 ${EXEC} -output ${RESULT_DIR}/32c_2s_smt.csv
numactl -C0-31,56-87,112-143,168-199 ${EXEC} -output ${RESULT_DIR}/64c_2s_smt.csv
# numactl -C0-55,56-111,112-167,168-223 ${EXEC} -output ${RESULT_DIR}/112c_2s_smt.csv
