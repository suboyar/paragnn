#!/usr/bin/env bash

#SBATCH --job-name="paragnn_benchmark"
#SBATCH --exclusive
#SBATCH --output=%j_%N_%x_output.txt

./build/${SLURM_JOB_PARTITION}/paragnn
