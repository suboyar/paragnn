#!/usr/bin/env bash

#SBATCH --job-name="compile-paragnn"
#SBATCH --time=0-00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%j_%N_%x_output.txt

./nob --rebuild --release --out-dir "${SLURM_JOB_PARTITION}"
