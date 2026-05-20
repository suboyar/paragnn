#!/usr/bin/env bash

#SBATCH --job-name="sageconv_backward_gemm_tn_v4"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive

set -euo pipefail

# send all informational/diagnostic output to stderr by default
exec 3>&1   # save original stdout as fd 3
exec 1>&2   # redirect stdout to stderr for everything below

if [ $(uname -m) = "x86_64" ];
then
    module purge
    module load gcc/cu1281/15.2.0
    module load openblas/dynamic/0.3.30
elif [ $(uname -m) = "aarch64" ];
then
    unset LOADEDMODULES
    unset _LMFILES_
    module purge
    module load gcc/cu1302/15.2.0
    module load openblas/dynamic/0.3.30
fi

TAG=$SLURM_JOB_PARTITION ./bootstrap.sh
NOB="./nob-$SLURM_JOB_PARTITION"
trap 'rm -f "$NOB"' EXIT

bench() {
    local dim=$1
    local kc=$((1 << $2))
    local mr=$((1 << $3))
    local nr=$((1 << $4))
    local rc out min_time

    $NOB -release -target sageconv_backward -o build/$SLURM_JOB_PARTITION \
         -D DIMS=$dim -D KC=$kc -D MR=$mr -D NR=$nr && rc=$? || rc=$?
    if [ $rc -ne 0 ]; then
        printf "dim=%d KC=%d MR=%d NR=%d compilation failed\n" "$dim" "$kc" "$mr" "$nr" >&2
        return 1
    fi

    out=$(./build/$SLURM_JOB_PARTITION/sageconv_backward 2>&1) && rc=$? || rc=$?

    if [ $rc -ne 0 ]; then
        printf "dim=%d KC=%d MR=%d NR=%d failed:\n%s\n" "$dim" "$kc" "$mr" "$nr" "$out" >&2
        return 1
    fi

    min_time=$(awk -F= '/^MIN_TIME/{print $2}' <<< "$out")
    if [[ -z "$min_time" ]]; then
        printf "dim=%d KC=%d MR=%d NR=%d failed:\n%s\n" "$dim" "$kc" "$mr" "$nr" "$out" >&2
        return 1
    fi

    printf "dim=%d KC=%d MR=%d NR=%d %s s\n" "$dim" "$kc" "$mr" "$nr" "$min_time" >&3
    if [[ -z "$best_time" ]] || awk "BEGIN{exit !($min_time < $best_time)}"; then
        best_time="$min_time"
        best_kc=$kc
        best_mr=$mr
        best_nr=$nr
    fi
}

for dim in 256 512 1024; do
    best_time=""
    best_kc=0
    best_mr=0
    best_nr=0
    for kc in {1..13}; do
        for mr in {1..6}; do
            for nr in {1..6}; do
                bench "$dim" "$kc" "$mr" "$nr" || true
            done
        done
    done
    if [[ -n "$best_time" ]]; then
        printf "Optimal Configuration with dim=%d is KC=%d MR=%d NR=%d %s s\n" "$dim" "$best_kc" "$best_mr" "$best_nr" "$best_time" >&3
    else
        printf "dim=%d: all runs failed\n" "$dim" >&2
    fi
done
