#!/usr/bin/env bash

#SBATCH --job-name="sageconv_backward_v5"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive


if [ "`uname -m`" = "x86_64" ];
then
    module purge
    module load gcc/cu1281/15.2.0 --silent
    module load openblas/dynamic/0.3.30 --silent
elif [ "`uname -m`" = "aarch64" ];
then
    unset LOADEDMODULES
    unset _LMFILES_
    module purge
    module load gcc/cu1302/15.2.0 --silent
    module load openblas/dynamic/0.3.30 --silent
fi

set -euo pipefail

best_time=""
best_jt=0
best_nb=0

TAG=$SLURM_JOB_PARTITION ./bootstrap.sh
NOB="./nob-$SLURM_JOB_PARTITION"

bench() {
    local jt=$1 nb=$2
    err=$($NOB -release -target sageconv_backward -o build/$SLURM_JOB_PARTITION \
               -D J_TILE=$((1 << jt)) -D SAGE_NODE_BLOCK=$((1 << nb)) 2>&1) || {
        printf "%s\n" "$err" >&2
        return 1
    }
    min_time=$(./build/$SLURM_JOB_PARTITION/sageconv_backward 2>&1 | awk -F= '/^MIN_V5/{print $2}')
    printf "J_TILE=%-5d  NODE_BLOCK=%-5d  %s s\n" \
           "$((1 << jt))" "$((1 << nb))" "$min_time" >&2
    if [[ -z "$best_time" ]] || awk "BEGIN{exit !($min_time < $best_time)}"; then
        best_time="$min_time"
        best_jt=$jt
        best_nb=$nb
    fi
}


for jt in {1..10}; do
    prev_time=""
    for nb in {1..10}; do
        bench "$jt" "$nb"
        if [[ -n "$prev_time" ]] && awk "BEGIN{exit !($min_time > $prev_time)}"; then
            printf "worse than previous NODE_BLOCK, skipping rest of J_TILE=%d\n" \
                   "$((1 << jt))" >&2
            break
        fi
        prev_time="$min_time"
    done
done


printf "Optimal Configuration J_TILE=%d  NODE_BLOCK=%d  %ss\n" \
       "$((1 << best_jt))" "$((1 << best_nb))" "$best_time"

rm $NOB
