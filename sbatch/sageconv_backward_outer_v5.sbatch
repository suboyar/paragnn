#!/usr/bin/env bash

#SBATCH --job-name="sageconv_backward_outer_v5"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive

set -euo pipefail

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


case $SLURM_JOB_PARTITION in
    armq)
        KC=(32 64 128 256 512)
        MR=(14  9  5  6  4)
        NR=( 8 12 20 16 24)
    ;;
    defq)
        KC=(32 64 128 256 512 1024 2048 4096 8192)
        MR=( 6  4)
        NR=(16 24)
        ;;
    fpgaq)
        KC=(32 64 128 256 512 1024 2048 4096 8192)
        MR=( 6  4)
        NR=(16 24)
        ;;
    genoaxq)
        KC=(32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536)
        MR=(14  9  5  6  4)
        NR=(32 48 80 64 96)
        ;;
    gh200q)
        KC=(32 64 128 256 512 1024 2048 4096 8192 16384)
        MR=(14  9  5  6  4)
        NR=( 8 12 20 16 24)
        ;;
    habanaq)
        KC=(32 64 128 256 512 1024 2048 4096 8192 16384 32768)
        MR=(14  9  5  6  4  3  2)
        NR=(32 48 80 64 96 112 160)
        ;;
    huaq)
        KC=(32 64 128 256 512 1024 2048 4096 8192 16384)
        MR=(14  9  5  6  4  3)
        NR=( 8 12 20 16 24 28)
        ;;
    milanq)
        KC=(32 64 128 256 512 1024 2048 4096 8192 16384)
        MR=( 6  4  2)
        NR=(16 24 40)
        ;;
    rome16q)
        KC=(32 64 128 256 512 1024 2048)
        MR=( 6  4)
        NR=(16 24)
        ;;
    xeonmaxq)
        KC=(32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536)
        MR=(14  9  5  6  4)
        NR=(32 48 80 64 96)
        ;;
    *)
        echo "Unknown partition: $SLURM_JOB_PARTITION" >&2
        exit 1
        ;;
esac

bench() {
    local kc=$1
    local mr=$2
    local nr=$3
    local rc out

    $NOB -release -target sageconv_backward -o build/$SLURM_JOB_PARTITION \
         -D KC=$kc -D MR=$mr -D NR=$nr && rc=$? || rc=$?
    if [ $rc -ne 0 ]; then
        printf "dim=%d KC=%d MR=%d NR=%d compilation failed\n" "$dim" "$kc" "$mr" "$nr" >&2
        return 1
    fi

    echo "--------------------------------------------------------------------------------"
    echo "Trying out KC=$kc MR=$mr NR=$nr"
    ./build/$SLURM_JOB_PARTITION/sageconv_backward && rc=$? || rc=$?

    if [ $rc -ne 0 ]; then
        printf "KC=%d MR=%d NR=%d failed:\n%s\n" "$kc" "$mr" "$nr" "$out" >&2
        return 1
    fi
    echo "--------------------------------------------------------------------------------"

}

for kc in "${KC[@]}"; do
    for i in "${!MR[@]}"; do
        bench "$kc" "${MR[$i]}" "${NR[$i]}" || true
    done
done
