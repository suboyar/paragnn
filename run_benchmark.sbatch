#!/usr/bin/env bash

#SBATCH --job-name="paragnn"
#SBATCH --exclusive
#SBATCH --output=logs/%j_%x.out

set -e

TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
RESULT_DIR="results/${SLURM_JOB_PARTITION}/${TIMESTAMP}"
BUILD_DIR=build/${SLURM_JOB_PARTITION}
EXEC=${BUILD_DIR}/paragnn

ARCH=$(uname -m)
NOB="./nob_${ARCH}"

if [ ! -f "${NOB}" ]; then
    gcc -o "${NOB}" nob.c
fi

${NOB} -rebuild -release -outdir ${BUILD_DIR}

mkdir -p ${RESULT_DIR}
module load numactl

function defq_affinity()
{
    # Single socket
    numactl -C0 ${EXEC} -csv ${RESULT_DIR}/01c_1s.csv

    numactl -C0-1 ${EXEC} -csv ${RESULT_DIR}/02c_1s.csv
    numactl -C0-3 ${EXEC} -csv ${RESULT_DIR}/04c_1s.csv
    numactl -C0-7 ${EXEC} -csv ${RESULT_DIR}/08c_1s.csv
    numactl -C0-15 ${EXEC} -csv ${RESULT_DIR}/16c_1s.csv
    numactl -C0-31 ${EXEC} -csv ${RESULT_DIR}/32c_1s.csv

    # Dual socket
    numactl -C0,32 ${EXEC} -csv ${RESULT_DIR}/02c_2s.csv
    numactl -C0-1,32-33 ${EXEC} -csv ${RESULT_DIR}/04c_2s.csv
    numactl -C0-3,32-35 ${EXEC} -csv ${RESULT_DIR}/08c_2s.csv
    numactl -C0-7,32-39 ${EXEC} -csv ${RESULT_DIR}/16c_2s.csv
    numactl -C0-15,32-47 ${EXEC} -csv ${RESULT_DIR}/32c_2s.csv
    numactl -C0-31,32-63 ${EXEC} -csv ${RESULT_DIR}/64c_2s.csv

    # Hyperthreading - Single socket
    numactl -C0,64 ${EXEC} -csv ${RESULT_DIR}/01c_1s_smt.csv
    numactl -C0-1,64-65 ${EXEC} -csv ${RESULT_DIR}/02c_1s_smt.csv
    numactl -C0-3,64-67 ${EXEC} -csv ${RESULT_DIR}/04c_1s_smt.csv
    numactl -C0-7,64-71 ${EXEC} -csv ${RESULT_DIR}/08c_1s_smt.csv
    numactl -C0-15,64-79 ${EXEC} -csv ${RESULT_DIR}/16c_1s_smt.csv
    numactl -C0-31,64-95 ${EXEC} -csv ${RESULT_DIR}/32c_1s_smt.csv

    # Hyperthreading - Dual socket
    numactl -C0,32,64,96 ${EXEC} -csv ${RESULT_DIR}/02c_2s_smt.csv
    numactl -C0-1,32-33,64-65,96-97 ${EXEC} -csv ${RESULT_DIR}/04c_2s_smt.csv
    numactl -C0-3,32-35,64-67,96-99 ${EXEC} -csv ${RESULT_DIR}/08c_2s_smt.csv
    numactl -C0-7,32-39,64-71,96-103 ${EXEC} -csv ${RESULT_DIR}/16c_2s_smt.csv
    numactl -C0-15,32-47,64-79,96-111 ${EXEC} -csv ${RESULT_DIR}/32c_2s_smt.csv
    numactl -C0-31,32-63,64-95,96-127 ${EXEC} -csv ${RESULT_DIR}/64c_2s_smt.csv
}

function fpgaq_affinity()
{
    # Single socket
    numactl -C0 ${EXEC} -csv ${RESULT_DIR}/01c_1s.csv
    numactl -C0-1 ${EXEC} -csv ${RESULT_DIR}/02c_1s.csv
    numactl -C0-3 ${EXEC} -csv ${RESULT_DIR}/04c_1s.csv
    numactl -C0-7 ${EXEC} -csv ${RESULT_DIR}/08c_1s.csv
    numactl -C0-15 ${EXEC} -csv ${RESULT_DIR}/16c_1s.csv
    # numactl -C0-23 ${EXEC} -csv ${RESULT_DIR}/24c_1s.csv

    # Dual socket
    numactl -C0,24 ${EXEC} -csv ${RESULT_DIR}/02c_2s.csv
    numactl -C0-1,24-25 ${EXEC} -csv ${RESULT_DIR}/04c_2s.csv
    numactl -C0-3,24-27 ${EXEC} -csv ${RESULT_DIR}/08c_2s.csv
    numactl -C0-7,24-31 ${EXEC} -csv ${RESULT_DIR}/16c_2s.csv
    numactl -C0-15,24-39 ${EXEC} -csv ${RESULT_DIR}/32c_2s.csv
    # numactl -C0-23,24-47 ${EXEC} -csv ${RESULT_DIR}/48c_2s.csv

    # Hyperthreading - Single socket
    numactl -C0,48 ${EXEC} -csv ${RESULT_DIR}/01c_1s_smt.csv
    numactl -C0-1,48-49 ${EXEC} -csv ${RESULT_DIR}/02c_1s_smt.csv
    numactl -C0-3,48-51 ${EXEC} -csv ${RESULT_DIR}/04c_1s_smt.csv
    numactl -C0-7,48-55 ${EXEC} -csv ${RESULT_DIR}/08c_1s_smt.csv
    numactl -C0-15,48-63 ${EXEC} -csv ${RESULT_DIR}/16c_1s_smt.csv
    # numactl -C0-23,48-71 ${EXEC} -csv ${RESULT_DIR}/24c_1s_smt.csv

    # Hyperthreading - Dual socket
    numactl -C0,24,48,72 ${EXEC} -csv ${RESULT_DIR}/02c_2s_smt.csv
    numactl -C0-1,24-25,48-49,72-73 ${EXEC} -csv ${RESULT_DIR}/04c_2s_smt.csv
    numactl -C0-3,24-27,48-51,72-75 ${EXEC} -csv ${RESULT_DIR}/08c_2s_smt.csv
    numactl -C0-7,24-31,48-55,72-79 ${EXEC} -csv ${RESULT_DIR}/16c_2s_smt.csv
    numactl -C0-15,24-39,48-63,72-87 ${EXEC} -csv ${RESULT_DIR}/32c_2s_smt.csv
    # numactl -C0-23,24-47,48-71,72-95 ${EXEC} -csv ${RESULT_DIR}/48c_2s_smt.csv
}

function genoaxq_affinity()
{
    # Single socket
    numactl -C0 ${EXEC} -csv ${RESULT_DIR}/01c_1s.csv
    numactl -C0-1 ${EXEC} -csv ${RESULT_DIR}/02c_1s.csv
    numactl -C0-3 ${EXEC} -csv ${RESULT_DIR}/04c_1s.csv
    numactl -C0-7 ${EXEC} -csv ${RESULT_DIR}/08c_1s.csv
    numactl -C0-15 ${EXEC} -csv ${RESULT_DIR}/16c_1s.csv
    numactl -C0-31 ${EXEC} -csv ${RESULT_DIR}/32c_1s.csv
    numactl -C0-63 ${EXEC} -csv ${RESULT_DIR}/64c_1s.csv
    # numactl -C0-95 ${EXEC} -csv ${RESULT_DIR}/96c_1s.csv

    # Dual socket
    numactl -C0,96 ${EXEC} -csv ${RESULT_DIR}/02c_2s.csv
    numactl -C0-1,96-97 ${EXEC} -csv ${RESULT_DIR}/04c_2s.csv
    numactl -C0-3,96-99 ${EXEC} -csv ${RESULT_DIR}/08c_2s.csv
    numactl -C0-7,96-103 ${EXEC} -csv ${RESULT_DIR}/16c_2s.csv
    numactl -C0-15,96-111 ${EXEC} -csv ${RESULT_DIR}/32c_2s.csv
    numactl -C0-31,96-127 ${EXEC} -csv ${RESULT_DIR}/64c_2s.csv
    numactl -C0-63,96-159 ${EXEC} -csv ${RESULT_DIR}/128c_2s.csv
    # numactl -C0-95,96-191 ${EXEC} -csv ${RESULT_DIR}/192c_2s.csv

    # Hyperthreading - Single socket
    numactl -C0,192 ${EXEC} -csv ${RESULT_DIR}/01c_1s_smt.csv
    numactl -C0-1,192-193 ${EXEC} -csv ${RESULT_DIR}/02c_1s_smt.csv
    numactl -C0-3,192-195 ${EXEC} -csv ${RESULT_DIR}/04c_1s_smt.csv
    numactl -C0-7,192-199 ${EXEC} -csv ${RESULT_DIR}/08c_1s_smt.csv
    numactl -C0-15,192-207 ${EXEC} -csv ${RESULT_DIR}/16c_1s_smt.csv
    numactl -C0-31,192-223 ${EXEC} -csv ${RESULT_DIR}/32c_1s_smt.csv
    numactl -C0-63,192-255 ${EXEC} -csv ${RESULT_DIR}/64c_1s_smt.csv
    # numactl -C0-95,192-287 ${EXEC} -csv ${RESULT_DIR}/96c_1s_smt.csv

    # Hyperthreading - Dual socket
    numactl -C0,96,192,288 ${EXEC} -csv ${RESULT_DIR}/02c_2s_smt.csv
    numactl -C0-1,96-97,192-193,288-289 ${EXEC} -csv ${RESULT_DIR}/04c_2s_smt.csv
    numactl -C0-3,96-99,192-195,288-291 ${EXEC} -csv ${RESULT_DIR}/08c_2s_smt.csv
    numactl -C0-7,96-103,192-199,288-295 ${EXEC} -csv ${RESULT_DIR}/16c_2s_smt.csv
    numactl -C0-15,96-111,192-207,288-303 ${EXEC} -csv ${RESULT_DIR}/32c_2s_smt.csv
    numactl -C0-31,96-127,192-223,288-319 ${EXEC} -csv ${RESULT_DIR}/64c_2s_smt.csv
    numactl -C0-63,96-159,192-255,288-351 ${EXEC} -csv ${RESULT_DIR}/128c_2s_smt.csv
    # numactl -C0-95,96-191,192-287,288-383 ${EXEC} -csv ${RESULT_DIR}/192c_2s_smt.csv
}

function gh200q_affinity()
{
    numactl -C0 ${EXEC} -csv ${RESULT_DIR}/01c.csv
    numactl -C0-1 ${EXEC} -csv ${RESULT_DIR}/02c.csv
    numactl -C0-3 ${EXEC} -csv ${RESULT_DIR}/04c.csv
    numactl -C0-7 ${EXEC} -csv ${RESULT_DIR}/08c.csv
    numactl -C0-15 ${EXEC} -csv ${RESULT_DIR}/16c.csv
    numactl -C0-31 ${EXEC} -csv ${RESULT_DIR}/32c.csv
    numactl -C0-63 ${EXEC} -csv ${RESULT_DIR}/64c.csv
    # numactl -C0-71 ${EXEC} -csv ${RESULT_DIR}/72c.csv
}

function milanq_affinity()
{
    # Single socket
    numactl -C0 ${EXEC} -csv ${RESULT_DIR}/01c_1s.csv
    numactl -C0-1 ${EXEC} -csv ${RESULT_DIR}/02c_1s.csv
    numactl -C0-3 ${EXEC} -csv ${RESULT_DIR}/04c_1s.csv
    numactl -C0-7 ${EXEC} -csv ${RESULT_DIR}/08c_1s.csv
    numactl -C0-15 ${EXEC} -csv ${RESULT_DIR}/16c_1s.csv
    numactl -C0-31 ${EXEC} -csv ${RESULT_DIR}/32c_1s.csv
    numactl -C0-63 ${EXEC} -csv ${RESULT_DIR}/64c_1s.csv

    # Dual socket
    numactl -C0,64 ${EXEC} -csv ${RESULT_DIR}/02c_2s.csv
    numactl -C0-1,64-65 ${EXEC} -csv ${RESULT_DIR}/04c_2s.csv
    numactl -C0-3,64-67 ${EXEC} -csv ${RESULT_DIR}/08c_2s.csv
    numactl -C0-7,64-71 ${EXEC} -csv ${RESULT_DIR}/16c_2s.csv
    numactl -C0-15,64-79 ${EXEC} -csv ${RESULT_DIR}/32c_2s.csv
    numactl -C0-31,64-95 ${EXEC} -csv ${RESULT_DIR}/64c_2s.csv
    numactl -C0-63,64-127 ${EXEC} -csv ${RESULT_DIR}/128c_2s.csv

    # Hyperthreading - Single socket
    numactl -C0,128 ${EXEC} -csv ${RESULT_DIR}/01c_1s_smt.csv
    numactl -C0-1,128-129 ${EXEC} -csv ${RESULT_DIR}/02c_1s_smt.csv
    numactl -C0-3,128-131 ${EXEC} -csv ${RESULT_DIR}/04c_1s_smt.csv
    numactl -C0-7,128-135 ${EXEC} -csv ${RESULT_DIR}/08c_1s_smt.csv
    numactl -C0-15,128-143 ${EXEC} -csv ${RESULT_DIR}/16c_1s_smt.csv
    numactl -C0-31,128-159 ${EXEC} -csv ${RESULT_DIR}/32c_1s_smt.csv
    numactl -C0-63,128-191 ${EXEC} -csv ${RESULT_DIR}/64c_1s_smt.csv

    # Hyperthreading - Dual socket
    numactl -C0,64,128,192 ${EXEC} -csv ${RESULT_DIR}/02c_2s_smt.csv
    numactl -C0-1,64-65,128-129,192-193 ${EXEC} -csv ${RESULT_DIR}/04c_2s_smt.csv
    numactl -C0-3,64-67,128-131,192-195 ${EXEC} -csv ${RESULT_DIR}/08c_2s_smt.csv
    numactl -C0-7,64-71,128-135,192-199 ${EXEC} -csv ${RESULT_DIR}/16c_2s_smt.csv
    numactl -C0-15,64-79,128-143,192-207 ${EXEC} -csv ${RESULT_DIR}/32c_2s_smt.csv
    numactl -C0-31,64-95,128-159,192-223 ${EXEC} -csv ${RESULT_DIR}/64c_2s_smt.csv
    numactl -C0-63,64-127,128-191,192-255 ${EXEC} -csv ${RESULT_DIR}/128c_2s_smt.csv
}

function xeonmaxq_affinity()
{
    # Single socket
    numactl -C0 ${EXEC} -csv ${RESULT_DIR}/01c_1s.csv
    numactl -C0-1 ${EXEC} -csv ${RESULT_DIR}/02c_1s.csv
    numactl -C0-3 ${EXEC} -csv ${RESULT_DIR}/04c_1s.csv
    numactl -C0-7 ${EXEC} -csv ${RESULT_DIR}/08c_1s.csv
    numactl -C0-15 ${EXEC} -csv ${RESULT_DIR}/16c_1s.csv
    numactl -C0-31 ${EXEC} -csv ${RESULT_DIR}/32c_1s.csv
    # numactl -C0-55 ${EXEC} -csv ${RESULT_DIR}/56c_1s.csv

    # Dual socket
    numactl -C0,56 ${EXEC} -csv ${RESULT_DIR}/02c_2s.csv
    numactl -C0-1,56-57 ${EXEC} -csv ${RESULT_DIR}/04c_2s.csv
    numactl -C0-3,56-59 ${EXEC} -csv ${RESULT_DIR}/08c_2s.csv
    numactl -C0-7,56-63 ${EXEC} -csv ${RESULT_DIR}/16c_2s.csv
    numactl -C0-15,56-71 ${EXEC} -csv ${RESULT_DIR}/32c_2s.csv
    numactl -C0-31,56-87 ${EXEC} -csv ${RESULT_DIR}/64c_2s.csv
    # numactl -C0-55,56-111 ${EXEC} -csv ${RESULT_DIR}/112c_2s.csv

    # Hyperthreading - Single socket
    numactl -C0,112 ${EXEC} -csv ${RESULT_DIR}/01c_1s_smt.csv
    numactl -C0-1,112-113 ${EXEC} -csv ${RESULT_DIR}/02c_1s_smt.csv
    numactl -C0-3,112-115 ${EXEC} -csv ${RESULT_DIR}/04c_1s_smt.csv
    numactl -C0-7,112-119 ${EXEC} -csv ${RESULT_DIR}/08c_1s_smt.csv
    numactl -C0-15,112-127 ${EXEC} -csv ${RESULT_DIR}/16c_1s_smt.csv
    numactl -C0-31,112-143 ${EXEC} -csv ${RESULT_DIR}/32c_1s_smt.csv
    # numactl -C0-55,112-167 ${EXEC} -csv ${RESULT_DIR}/56c_1s_smt.csv

    # Hyperthreading - Dual socket
    numactl -C0,56,112,168 ${EXEC} -csv ${RESULT_DIR}/02c_2s_smt.csv
    numactl -C0-1,56-57,112-113,168-169 ${EXEC} -csv ${RESULT_DIR}/04c_2s_smt.csv
    numactl -C0-3,56-59,112-115,168-171 ${EXEC} -csv ${RESULT_DIR}/08c_2s_smt.csv
    numactl -C0-7,56-63,112-119,168-175 ${EXEC} -csv ${RESULT_DIR}/16c_2s_smt.csv
    numactl -C0-15,56-71,112-127,168-183 ${EXEC} -csv ${RESULT_DIR}/32c_2s_smt.csv
    numactl -C0-31,56-87,112-143,168-199 ${EXEC} -csv ${RESULT_DIR}/64c_2s_smt.csv
    # numactl -C0-55,56-111,112-167,168-223 ${EXEC} -csv ${RESULT_DIR}/112c_2s_smt.csv
}

function rome16q_affinity()
{
    # Single socket
    numactl -C0 ${EXEC} -csv ${RESULT_DIR}/01c.csv
    numactl -C0-1 ${EXEC} -csv ${RESULT_DIR}/02c.csv
    numactl -C0-3 ${EXEC} -csv ${RESULT_DIR}/04c.csv
    numactl -C0-7 ${EXEC} -csv ${RESULT_DIR}/08c.csv
    numactl -C0-15 ${EXEC} -csv ${RESULT_DIR}/16c.csv
    
    # Hyperthreading - Single socket
    numactl -C0,16 ${EXEC} -csv ${RESULT_DIR}/01c_smt.csv
    numactl -C0-1,16-17 ${EXEC} -csv ${RESULT_DIR}/02c_smt.csv
    numactl -C0-3,16-19 ${EXEC} -csv ${RESULT_DIR}/04c_smt.csv
    numactl -C0-7,16-23 ${EXEC} -csv ${RESULT_DIR}/08c_smt.csv
    numactl -C0-15,16-31 ${EXEC} -csv ${RESULT_DIR}/16c_smt.csv
}

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
