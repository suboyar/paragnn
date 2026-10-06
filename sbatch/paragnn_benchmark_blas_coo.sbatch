#!/usr/bin/env bash

#SBATCH --job-name="paragnn_benchmark_blas_coo"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive

if [ "$(uname -m)" = "x86_64" ];
then
    module purge
    module load gcc/cu1281/15.2.0
    module load openblas/dynamic/0.3.30
    module load numactl/gcc/2.0.17
elif [ "$(uname -m)" = "aarch64" ];
then
    unset LOADEDMODULES
    unset _LMFILES_
    module purge
    module load gcc/cu1302/15.2.0
    module load openblas/dynamic/0.3.30
    module load numactl/gcc/2.0.14
fi

run_cmd() {
    echo "+ $*" >&2
    "$@"
}

count_threads() {
    local count=0
    local IFS=','
    for part in $1; do
        if [[ $part == *-* ]]; then
            (( count += ${part#*-} - ${part%-*} + 1 ))
        else
            (( count += 1 ))
        fi
    done
    echo "$count"
}

BDIR="build_${SLURM_JOB_PARTITION}"

run_cmd ulimit -v unlimited
run_cmd make -B paragnn BUILDDIR="$BDIR" MODE=benchmark IMPL=blas SPARSE=cs V=1 1>&2

OUTPUT_DIR="${OUTPUT_DIR:-results}"
mkdir -p "$OUTPUT_DIR"

run_fixed() {
    local dataset="$1"
    for mode in "1s" "1s-smt" "2s" "2s-smt"; do
        local c=$(./cpumap ${mode})
        local t=$(count_threads "$c")
        local run_label="${mode}-${t}t"

        echo
        echo "----------------------------------------------------------------"
        echo " ${dataset} (${run_label})"
        echo "----------------------------------------------------------------"
        run_cmd env OMP_NUM_THREADS="$t" OPENBLAS_NUM_THREADS="$t" OMP_PROC_BIND=true OMP_PLACES=threads numactl -C"$c" \
                ./"$BDIR"/paragnn --dataset "$dataset" --csv --output "${OUTPUT_DIR}/${SLURM_JOB_PARTITION}_${dataset}_${run_label}.csv"
    done
}

run_scaled() {
    local dataset="$1"
    for mode in "scale-1s" "scale-1s-smt" "scale-2s" "scale-2s-smt"; do
        for c in $(./cpumap ${mode}); do
            local t=$(count_threads "$c")
            local run_label="${mode#scale-}-${t}t"
            echo
            echo "----------------------------------------------------------------"
            echo " ${dataset} (${run_label})"
            echo "----------------------------------------------------------------"
            run_cmd env OMP_NUM_THREADS="$t" OPENBLAS_NUM_THREADS="$t" OMP_PROC_BIND=true OMP_PLACES=threads numactl -C"$c" \
                    ./"$BDIR"/paragnn --dataset "$dataset" --csv --output "${OUTPUT_DIR}/${SLURM_JOB_PARTITION}_${dataset}_${run_label}.csv"
        done
    done
}

case "$SLURM_JOB_PARTITION" in
    gh200q|h200q|genoaxq)
        run_scaled "ogbn-arxiv"
        run_scaled "ogbn-products"
        run_fixed "ogbn-papers100M"
        ;;
    *)
        run_fixed "ogbn-arxiv"
        run_fixed "ogbn-products"
        run_fixed "ogbn-papers100M"
        ;;
esac
