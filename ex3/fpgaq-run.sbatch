#!/usr/bin/env bash

#SBATCH --job-name="paragnn_benchmark"
#SBATCH --exclusive
#SBATCH --partition=fpgaq
#SBATCH --output=fpgaq-%j-%x-stdout.txt
#SBATCH --error=fpgaq-%j-%x-stderr.txt

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
# numactl -C0-23 ${EXEC} -output ${RESULT_DIR}/24c_1s.csv

# Dual socket
numactl -C0,24 ${EXEC} -output ${RESULT_DIR}/2c_2s.csv
numactl -C0-1,24-25 ${EXEC} -output ${RESULT_DIR}/4c_2s.csv
numactl -C0-3,24-27 ${EXEC} -output ${RESULT_DIR}/8c_2s.csv
numactl -C0-7,24-31 ${EXEC} -output ${RESULT_DIR}/16c_2s.csv
numactl -C0-15,24-39 ${EXEC} -output ${RESULT_DIR}/32c_2s.csv
# numactl -C0-23,24-47 ${EXEC} -output ${RESULT_DIR}/48c_2s.csv

# Hyperthreading - Single socket
numactl -C0,48 ${EXEC} -output ${RESULT_DIR}/1c_1s_smt.csv
numactl -C0-1,48-49 ${EXEC} -output ${RESULT_DIR}/2c_1s_smt.csv
numactl -C0-3,48-51 ${EXEC} -output ${RESULT_DIR}/4c_1s_smt.csv
numactl -C0-7,48-55 ${EXEC} -output ${RESULT_DIR}/8c_1s_smt.csv
numactl -C0-15,48-63 ${EXEC} -output ${RESULT_DIR}/16c_1s_smt.csv
# numactl -C0-23,48-71 ${EXEC} -output ${RESULT_DIR}/24c_1s_smt.csv

# Hyperthreading - Dual socket
numactl -C0,24,48,72 ${EXEC} -output ${RESULT_DIR}/2c_2s_smt.csv
numactl -C0-1,24-25,48-49,72-73 ${EXEC} -output ${RESULT_DIR}/4c_2s_smt.csv
numactl -C0-3,24-27,48-51,72-75 ${EXEC} -output ${RESULT_DIR}/8c_2s_smt.csv
numactl -C0-7,24-31,48-55,72-79 ${EXEC} -output ${RESULT_DIR}/16c_2s_smt.csv
numactl -C0-15,24-39,48-63,72-87 ${EXEC} -output ${RESULT_DIR}/32c_2s_smt.csv
# numactl -C0-23,24-47,48-71,72-95 ${EXEC} -output ${RESULT_DIR}/48c_2s_smt.csv
