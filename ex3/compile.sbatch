#!/usr/bin/env bash

#SBATCH --job-name="compile-paragnn"
#SBATCH --time=0-00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/compile/%j_%N_%x_output.txt

set -xe

# EXPERIMENT is passed via --export from nob.c
EXPERIMENT="${EXPERIMENT:-baseline}"

sleep 5s

./nob -build -force-rebuild -release -experiment "${EXPERIMENT}" -partition "${SLURM_JOB_PARTITION}"
