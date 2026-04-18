#!/usr/bin/env bash

#SBATCH --job-name="sageconv_backward_gemm_tn_v2"
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
    local kc=$2
    local rc out min_time

    $NOB -release -target sageconv_backward -o build/$SLURM_JOB_PARTITION \
         -D KC=$((1 << kc)) -D DIMS=$dim && rc=$? || rc=$?
    if [ $rc -ne 0 ]; then
        printf "dim=%d KC=%d compilation failed\n" "$dim" "$kc" >&2
        return 1
    fi

    out=$(./build/$SLURM_JOB_PARTITION/sageconv_backward 2>&1) && rc=$? || rc=$?

    if [ $rc -ne 0 ]; then
        printf "dim=%d KC=%d failed:\n%s\n" "$dim" "$kc" "$out" >&2
        return 1
    fi

    min_time=$(awk -F= '/^MIN_TIME/{print $2}' <<< "$out")
    if [[ -z "$min_time" ]]; then
        printf "dim=%d KC=%d no MIN_TIME\n" "$dim" "$kc" >&2
        return 1
    fi

    printf "dim=%d KC=%d  %s s\n" "$dim" "$((1 << kc))" "$min_time" >&3
    if [[ -z "$best_time" ]] || awk "BEGIN{exit !($min_time < $best_time)}"; then
        best_time="$min_time"
        best_kc=$kc
    fi
}

for dim in 256 512 1024; do
    best_time=""
    best_kc=0
    for kc in {1..13}; do
        bench "$dim" "$kc" || true
    done
    if [[ -n "$best_time" ]]; then
        printf "Optimal Configuration with dim=%d is KC=%d  %s s\n" "$dim" "$((1 << best_kc))" "$best_time" >&3
    else
        printf "dim=%d: all runs failed\n" "$dim"
    fi
done
