#!/usr/bin/env bash

#SBATCH --job-name="paragnn_benchmark"
#SBATCH --exclusive
#SBATCH --partition=milanq
#SBATCH --output=milanq-%j-%x-stdout.txt
#SBATCH --error=milanq-%j-%x-stderr.txt

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
numactl -C0-63 ${EXEC} -output ${RESULT_DIR}/64c_1s.csv

# Dual socket
numactl -C0,64 ${EXEC} -output ${RESULT_DIR}/2c_2s.csv
numactl -C0-1,64-65 ${EXEC} -output ${RESULT_DIR}/4c_2s.csv
numactl -C0-3,64-67 ${EXEC} -output ${RESULT_DIR}/8c_2s.csv
numactl -C0-7,64-71 ${EXEC} -output ${RESULT_DIR}/16c_2s.csv
numactl -C0-15,64-79 ${EXEC} -output ${RESULT_DIR}/32c_2s.csv
numactl -C0-31,64-95 ${EXEC} -output ${RESULT_DIR}/64c_2s.csv
numactl -C0-63,64-127 ${EXEC} -output ${RESULT_DIR}/128c_2s.csv

# Hyperthreading - Single socket
numactl -C0,128 ${EXEC} -output ${RESULT_DIR}/1c_1s_smt.csv
numactl -C0-1,128-129 ${EXEC} -output ${RESULT_DIR}/2c_1s_smt.csv
numactl -C0-3,128-131 ${EXEC} -output ${RESULT_DIR}/4c_1s_smt.csv
numactl -C0-7,128-135 ${EXEC} -output ${RESULT_DIR}/8c_1s_smt.csv
numactl -C0-15,128-143 ${EXEC} -output ${RESULT_DIR}/16c_1s_smt.csv
numactl -C0-31,128-159 ${EXEC} -output ${RESULT_DIR}/32c_1s_smt.csv
numactl -C0-63,128-191 ${EXEC} -output ${RESULT_DIR}/64c_1s_smt.csv

# Hyperthreading - Dual socket
numactl -C0,64,128,192 ${EXEC} -output ${RESULT_DIR}/2c_2s_smt.csv
numactl -C0-1,64-65,128-129,192-193 ${EXEC} -output ${RESULT_DIR}/4c_2s_smt.csv
numactl -C0-3,64-67,128-131,192-195 ${EXEC} -output ${RESULT_DIR}/8c_2s_smt.csv
numactl -C0-7,64-71,128-135,192-199 ${EXEC} -output ${RESULT_DIR}/16c_2s_smt.csv
numactl -C0-15,64-79,128-143,192-207 ${EXEC} -output ${RESULT_DIR}/32c_2s_smt.csv
numactl -C0-31,64-95,128-159,192-223 ${EXEC} -output ${RESULT_DIR}/64c_2s_smt.csv
numactl -C0-63,64-127,128-191,192-255 ${EXEC} -output ${RESULT_DIR}/128c_2s_smt.csv
