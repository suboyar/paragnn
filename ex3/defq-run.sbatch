#!/usr/bin/env bash

#SBATCH --job-name="paragnn_benchmark"
#SBATCH --exclusive
#SBATCH --partition=defq
#SBATCH --output=defq-%j-%x-stdout.txt
#SBATCH --error=defq-%j-%x-stderr.txt

set -e

# EXPERIMENT is passed via --export from nob.c
EXPERIMENT="${EXPERIMENT:-baseline}"

TIMESTAMP=$(date +%Y%m%dT%H%M%S)
RESULT_DIR="results/${EXPERIMENT}/${SLURM_JOB_PARTITION}/${TIMESTAMP}"
EXEC=./build/${EXPERIMENT}/${SLURM_JOB_PARTITION}/paragnn

module load numactl

# Single socket
numactl -C0 ${EXEC} -output ${RESULT_DIR}/1c_1s.csv

numactl -C0-1 ${EXEC} -output ${RESULT_DIR}/2c_1s.csv
numactl -C0-3 ${EXEC} -output ${RESULT_DIR}/4c_1s.csv
numactl -C0-7 ${EXEC} -output ${RESULT_DIR}/8c_1s.csv
numactl -C0-15 ${EXEC} -output ${RESULT_DIR}/16c_1s.csv
numactl -C0-31 ${EXEC} -output ${RESULT_DIR}/32c_1s.csv

# Dual socket
numactl -C0,32 ${EXEC} -output ${RESULT_DIR}/2c_2s.csv
numactl -C0-1,32-33 ${EXEC} -output ${RESULT_DIR}/4c_2s.csv
numactl -C0-3,32-35 ${EXEC} -output ${RESULT_DIR}/8c_2s.csv
numactl -C0-7,32-39 ${EXEC} -output ${RESULT_DIR}/16c_2s.csv
numactl -C0-15,32-47 ${EXEC} -output ${RESULT_DIR}/32c_2s.csv
numactl -C0-31,32-63 ${EXEC} -output ${RESULT_DIR}/64c_2s.csv

# Hyperthreading - Single socket
numactl -C0,64 ${EXEC} -output ${RESULT_DIR}/1c_1s_smt.csv
numactl -C0-1,64-65 ${EXEC} -output ${RESULT_DIR}/2c_1s_smt.csv
numactl -C0-3,64-67 ${EXEC} -output ${RESULT_DIR}/4c_1s_smt.csv
numactl -C0-7,64-71 ${EXEC} -output ${RESULT_DIR}/8c_1s_smt.csv
numactl -C0-15,64-79 ${EXEC} -output ${RESULT_DIR}/16c_1s_smt.csv
numactl -C0-31,64-95 ${EXEC} -output ${RESULT_DIR}/32c_1s_smt.csv

# Hyperthreading - Dual socket
numactl -C0,32,64,96 ${EXEC} -output ${RESULT_DIR}/2c_2s_smt.csv
numactl -C0-1,32-33,64-65,96-97 ${EXEC} -output ${RESULT_DIR}/4c_2s_smt.csv
numactl -C0-3,32-35,64-67,96-99 ${EXEC} -output ${RESULT_DIR}/8c_2s_smt.csv
numactl -C0-7,32-39,64-71,96-103 ${EXEC} -output ${RESULT_DIR}/16c_2s_smt.csv
numactl -C0-15,32-47,64-79,96-111 ${EXEC} -output ${RESULT_DIR}/32c_2s_smt.csv
numactl -C0-31,32-63,64-95,96-127 ${EXEC} -output ${RESULT_DIR}/64c_2s_smt.csv
