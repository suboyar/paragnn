#!/usr/bin/env bash

#SBATCH --job-name="paragnn_benchmark"
#SBATCH --exclusive
#SBATCH --partition=genoaxq
#SBATCH --output=genoaxq-%j-%x-stdout.txt
#SBATCH --error=genoaxq-%j-%x-stderr.txt

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
# numactl -C0-95 ${EXEC} -output ${RESULT_DIR}/96c_1s.csv

# Dual socket
numactl -C0,96 ${EXEC} -output ${RESULT_DIR}/2c_2s.csv
numactl -C0-1,96-97 ${EXEC} -output ${RESULT_DIR}/4c_2s.csv
numactl -C0-3,96-99 ${EXEC} -output ${RESULT_DIR}/8c_2s.csv
numactl -C0-7,96-103 ${EXEC} -output ${RESULT_DIR}/16c_2s.csv
numactl -C0-15,96-111 ${EXEC} -output ${RESULT_DIR}/32c_2s.csv
numactl -C0-31,96-127 ${EXEC} -output ${RESULT_DIR}/64c_2s.csv
numactl -C0-63,96-159 ${EXEC} -output ${RESULT_DIR}/128c_2s.csv
# numactl -C0-95,96-191 ${EXEC} -output ${RESULT_DIR}/192c_2s.csv

# Hyperthreading - Single socket
numactl -C0,192 ${EXEC} -output ${RESULT_DIR}/1c_1s_smt.csv
numactl -C0-1,192-193 ${EXEC} -output ${RESULT_DIR}/2c_1s_smt.csv
numactl -C0-3,192-195 ${EXEC} -output ${RESULT_DIR}/4c_1s_smt.csv
numactl -C0-7,192-199 ${EXEC} -output ${RESULT_DIR}/8c_1s_smt.csv
numactl -C0-15,192-207 ${EXEC} -output ${RESULT_DIR}/16c_1s_smt.csv
numactl -C0-31,192-223 ${EXEC} -output ${RESULT_DIR}/32c_1s_smt.csv
numactl -C0-63,192-255 ${EXEC} -output ${RESULT_DIR}/64c_1s_smt.csv
# numactl -C0-95,192-287 ${EXEC} -output ${RESULT_DIR}/96c_1s_smt.csv

# Hyperthreading - Dual socket
numactl -C0,96,192,288 ${EXEC} -output ${RESULT_DIR}/2c_2s_smt.csv
numactl -C0-1,96-97,192-193,288-289 ${EXEC} -output ${RESULT_DIR}/4c_2s_smt.csv
numactl -C0-3,96-99,192-195,288-291 ${EXEC} -output ${RESULT_DIR}/8c_2s_smt.csv
numactl -C0-7,96-103,192-199,288-295 ${EXEC} -output ${RESULT_DIR}/16c_2s_smt.csv
numactl -C0-15,96-111,192-207,288-303 ${EXEC} -output ${RESULT_DIR}/32c_2s_smt.csv
numactl -C0-31,96-127,192-223,288-319 ${EXEC} -output ${RESULT_DIR}/64c_2s_smt.csv
numactl -C0-63,96-159,192-255,288-351 ${EXEC} -output ${RESULT_DIR}/128c_2s_smt.csv
# numactl -C0-95,96-191,192-287,288-383 ${EXEC} -output ${RESULT_DIR}/192c_2s_smt.csv
