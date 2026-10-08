#!/usr/bin/env bash

#SBATCH --job-name="bench_agg_benchmark"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive


if [ $(uname -m) = "x86_64" ];
then
    module purge
    module load gcc/cu1281/15.2.0
    module load openblas/dynamic/0.3.30
    module load numactl/gcc/2.0.17
elif [ $(uname -m) = "aarch64" ];
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

get_label_cores() {
    local mode=$1
    local t=$2
    case "$mode" in
        *2s-smt) echo "$(( t / 4 ))" ;;
        *1s-smt|*2s) echo "$(( t / 2 ))" ;;
        *) echo "$t" ;;
    esac
}

run_cmd ulimit -v unlimited

BASE_DIR="${RESULTS_DIR:-results}"
PREFIX_DIR="${BASE_DIR}/${SLURM_JOB_PARTITION}"
mkdir -p "${PREFIX_DIR}-kernel" "${PREFIX_DIR}-complete" \
      "${PREFIX_DIR}-kernel-ideg" "${PREFIX_DIR}-complete-ideg"

BDIR="build_${SLURM_JOB_PARTITION}"

case "$SLURM_JOB_PARTITION" in
    # gh200q|h200q|genoaxq) MODES=("scale-1s" "scale-1s-smt" "scale-2s" "scale-2s-smt") ;;
    h200q|genoaxq) MODES=("scale-1s" "scale-1s-smt" "scale-2s" "scale-2s-smt") ;;
    *)                    MODES=("1s" "1s-smt" "2s" "2s-smt") ;;
esac

DATASETS=("ogbn-arxiv" "ogbn-products" "ogbn-papers100M")
DATASETS=("ogbn-arxiv")

run_suite() {
    local allocation="$1"
    local run_mode="$2"
    local dir="${PREFIX_DIR}-${run_mode}"

    for dataset in "${DATASETS[@]}"; do
        for mode in "${MODES[@]}"; do
            for c in $(./cpumap "$mode"); do
                local t=$(count_threads "$c")
                local label_t=$(get_label_cores "$mode" "$t")
                local run_label="${mode#scale-}-${label_t}t"

                echo
                echo "----------------------------------------------------------------"
                echo " ${dataset} ${run_mode} ${allocation} (${run_label})"
                echo "----------------------------------------------------------------"
                run_cmd env OMP_NUM_THREADS="$t" OPENBLAS_NUM_THREADS="$t" \
                    OMP_PROC_BIND=true OMP_PLACES=threads numactl -C"$c" \
                    ./"$BDIR"/bench-agg --dataset "$dataset" --csv \
                    --output-timer "${dir}/${dataset}_${allocation}_timer_${run_label}.csv" \
                    --output-perf "${dir}/${dataset}_${allocation}_perf_${run_label}.csv" \
                    --ntimes 1
            done
        done
    done
}

run_cmd make -B bench-agg BUILDDIR="$BDIR" V=1 1>&2
run_suite "interleave" "kernel"

run_cmd make -B bench-agg BUILDDIR="$BDIR" CFLAGS="-DFIRST_TOUCH" V=1 1>&2
run_suite "first-touch" "kernel"

run_cmd make -B bench-agg BUILDDIR="$BDIR" CFLAGS="-DCACHE_INV_DEGREE"V=1 1>&2
run_suite "interleave" "kernel-idge"

run_cmd make -B bench-agg BUILDDIR="$BDIR" CFLAGS="-DFIRST_TOUCH -DCACHE_INV_DEGREE" V=1 1>&2
run_suite "first-touch" "kernel-ideg"


run_cmd make -B bench-agg BUILDDIR="$BDIR" CFLAGS="-DRUN_COMPLETE_GRAD_SAGECONV" V=1 1>&2
run_suite "interleave" "complete"

run_cmd make -B bench-agg BUILDDIR="$BDIR" CFLAGS="-DRUN_COMPLETE_GRAD_SAGECONV -DFIRST_TOUCH" V=1 1>&2
run_suite "first-touch" "complete"

run_cmd make -B bench-agg BUILDDIR="$BDIR" CFLAGS="-DRUN_COMPLETE_GRAD_SAGECONV -DCACHE_INV_DEGREE" V=1 1>&2
run_suite "interleave" "complete-ideg"

run_cmd make -B bench-agg BUILDDIR="$BDIR" CFLAGS="-DRUN_COMPLETE_GRAD_SAGECONV -DFIRST_TOUCH -DCACHE_INV_DEGREE" V=1 1>&2
run_suite "first-touch" "complete-ideg"
