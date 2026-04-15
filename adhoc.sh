#!/usr/bin/env bash

#SBATCH --job-name="sageconv_backward_v4"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive


if [ "`uname -m`" = "x86_64" ];
then
    module purge
    module load gcc/cu1281/15.2.0
    module load openblas/dynamic/0.3.30
elif [ "`uname -m`" = "aarch64" ];
then
    unset LOADEDMODULES
    unset _LMFILES_
    module purge
    module load gcc/cu1302/15.2.0
    module load openblas/dynamic/0.3.30
fi

set -euo pipefail

best_time=""
best_nb=0

TAG=$SLURM_JOB_PARTITION ./bootstrap.sh
NOB="./nob-$SLURM_JOB_PARTITION"

bench() {
    local nb=$1
    err=$($NOB -release -target sageconv_backward -o build/$SLURM_JOB_PARTITION \
               -D SAGE_NODE_BLOCK=$((1 << nb)) 2>&1) || {
        printf "%s\n" "$err" >&2
        return 1
    }
    min_time=$(./build/$SLURM_JOB_PARTITION/sageconv_backward 2>&1 | awk -F= '/^MIN_V4/{print $2}')
    printf "NODE_BLOCK=%-5d  %s s\n" \
            "$((1 << nb))" "$min_time" >&2
    if [[ -z "$best_time" ]] || awk "BEGIN{exit !($min_time < $best_time)}"; then
        best_time="$min_time"
        best_nb=$nb
    fi
}

for nb in {1..10}; do
    bench "$nb"
done

printf "Optimal Configuration  NODE_BLOCK=%d  %s s\n" \
       "$((1 << best_nb))" "$best_time"

rm $NOB
