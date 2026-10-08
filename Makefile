.DELETE_ON_ERROR:

MAKEFLAGS += -j$(shell nproc)

PARTITION ?= $(or $(SLURM_JOB_PARTITION),default)
-include mkconfigs/$(PARTITION).mk

# Defaulst
CC         ?= gcc
CFLAGS     ?=
LDFLAGS    ?=
LDLIBS     ?=
DEBUG      ?= 0
OPENMP     ?= 1
USE_DOUBLE ?= 0
MODE       ?= convergence
IMPL       ?= blas
SPARSE     ?= cs
BUILDDIR   ?= build
TARGET_CPU ?= TARGET_CPU_GENERIC
MARCH      ?= native
V          ?= 0
DATADIR ?= ~/D1/paragnn-dataset

# remove trailing slash if included
BUILDDIR := $(patsubst %/,%,$(BUILDDIR))

DEFAULT_CFLAGS += -std=gnu23
DEFAULT_CFLAGS += -Wall \
                  -Wextra \
                  -Wfloat-conversion \
                  -Werror=implicit-function-declaration \
                  -Werror=strict-prototypes \
                  -Werror=incompatible-pointer-types \
                  -Wno-unused-function\
                  -ggdb

DEFAULT_CFLAGS += -D$(TARGET_CPU)
# Add dummy targets for local header files
DEFAULT_CFLAGS += -MMD -MP

DEFAULT_LDLIBS = -lm -lopenblas -lnuma

ifeq ($(DEBUG),1)
    DEFAULT_CFLAGS += -O0 -g3 -gdwarf-2 -march=$(MARCH)
    # suppress ABI warnings from platform-specific vector types
    DEFAULT_CFLAGS += -Wno-psabi
else
    DEFAULT_CFLAGS += -O3 -ffast-math -march=$(MARCH) -DNDEBUG
endif

ifeq ($(OPENMP),1)
    DEFAULT_CFLAGS += -fopenmp
else
    DEFAULT_LDLIBS += -lgomp
    DEFAULT_CFLAGS += -Wno-unknown-pragmas
endif

ifeq ($(USE_DOUBLE),1)
    DEFAULT_CFLAGS += -DUSE_DOUBLE
endif

ifeq ($(MODE),benchmark)
    DEFAULT_CFLAGS += -DBENCHMARK_MODE
endif

ifeq ($(IMPL),naive)
    DEFAULT_CFLAGS += -DSAGECONV_NAIVE_IMPL
else ifeq ($(IMPL),blas)
    DEFAULT_CFLAGS += -DSAGECONV_BLAS_IMPL
else
    DEFAULT_CFLAGS += -DSAGECONV_TUNED_IMPL
endif

ifeq ($(SPARSE),coo)
    DEFAULT_CFLAGS += -DSPARSE_COO
endif

ALL_CFLAGS = $(strip $(DEFAULT_CFLAGS) $(CFLAGS))
ALL_LDFLAGS = $(strip $(DEFAULT_LDFLAGS) $(LDFLAGS))
ALL_LDLIBS = $(strip $(DEFAULT_LDLIBS) $(LDLIBS))

to_obj = $(patsubst %.c,$(BUILDDIR)/%.o,$1)
to_bench_obj = $(patsubst %.c,$(BENCHDIR)/%.o,$1)

PARAGNN_SRCS = src/main.c \
               src/core.c \
               src/ds.c \
               src/dsinfo.c \
               src/layers.c \
               src/matmul_naive.c \
               src/nn.c \
               src/optim.c \
               src/sageconv.c \
               src/sparsegraph.c \
               src/timer.c

BENCH_GS_SRCS := benchmark/grad_sageconv/main.c \
                 benchmark/grad_sageconv/naive.c \
                 benchmark/grad_sageconv/blas.c \
                 benchmark/grad_sageconv/outer_tn_v1.c \
                 benchmark/grad_sageconv/outer_tn_v2.c \
                 benchmark/grad_sageconv/outer_tn_v3.c \
                 benchmark/membw.c \
                 src/core.c \
                 src/ds.c \
                 src/dsinfo.c \
                 src/layers.c \
                 src/sparsegraph.c \
                 src/timer.c

AGGREGATE_SRCS := benchmark/mean_aggregate/main.c \
                  benchmark/mean_aggregate/coo_v1.c \
                  benchmark/mean_aggregate/cs_v1.c \
                  benchmark/mean_aggregate/cs_v2.c \
                  benchmark/mean_aggregate/cs_v3.c \
                  benchmark/mean_aggregate/cs_v4.c \
                  benchmark/membw.c \
                  src/core.c \
                  src/ds.c \
                  src/dsinfo.c \
                  src/sparsegraph.c \
                  src/timer.c

DSPREP_SRC :=  src/dsprep.c src/dsinfo.c src/core.c

all: paragnn bench-gs dsprep

ifeq ($(V),1)
    Q =
    E = @true
else
    Q = @
    E = @echo
endif

paragnn: $(patsubst %.c,$(BUILDDIR)/%.o,$(PARAGNN_SRCS))
	$(E) "  LD    $@"
	$(Q)$(CC) $(ALL_CFLAGS) $(ALL_LDFLAGS) -o $(BUILDDIR)/$@ $^ $(ALL_LDLIBS)

bench-gs: $(patsubst %.c,$(BUILDDIR)/%.o,$(BENCH_GS_SRCS))
	$(E) "  LD    $@"
	$(Q)$(CC) $(ALL_CFLAGS) $(ALL_LDFLAGS) -o $(BUILDDIR)/$@ $^ $(ALL_LDLIBS)

bench-agg: $(call to_obj,$(AGGREGATE_SRCS)) | $(BUILDDIR)
	$(E) "  LD    $@"
	$(Q)$(CC) $(ALL_CFLAGS) $(ALL_LDFLAGS) -o $(BUILDDIR)/$@ $^ $(ALL_LDLIBS)

# Some invalid values might be nan or inf, and they will be converted to -1, but
# for that to happen we need to disable -ffast-math's -ffinite-math-only
dsprep: ALL_CFLAGS += -fno-finite-math-only
dsprep: $(call to_obj,$(DSPREP_SRC)) | $(BUILDDIR)
	$(E) "  LD    $@"
	$(Q)$(CC) $(ALL_CFLAGS) $(ALL_LDFLAGS) -o $(BUILDDIR)/$@ $^ $(ALL_LDLIBS) -lz

$(BUILDDIR)/%.o: %.c
	@mkdir -p $(dir $@)
	$(E) "  CC    $<"
	$(Q)$(CC) $(ALL_CFLAGS) -Isrc/ -c $< -o $@

$(BUILDDIR)/%.o: %.c
	@mkdir -p $(dir $@)
	$(E) "  CC    $<"
	$(Q)$(CC) $(ALL_CFLAGS) -Isrc/ -c $< -o $@

ifeq ($(V),1)
    DS_VERBOSE = --verbose
endif

ogbn-arxiv ogbn-products ogbn-papers100M: $(BUILDDIR)/dsprep
	./$< --dataset $@ --root $(DATADIR) $(DS_VERBOSE)

datasets: $(BUILDDIR)/dsprep
	./$< --dataset all --root $(DATADIR) $(DS_VERBOSE)

tags:
	$(E) "  Generating etags..."
	$(Q)rm -f TAGS
	$(Q)find src/ benchmark/ -type f -name '*.[ch]' -print0 | xargs -0 etags -a --declarations

clean:
	rm -rf $(BENCHDIR)

help:
	@echo "Usage: make [TARGET] [OPTIONS]"
	@echo ""
	@echo "Targets:"
	@echo "  paragnn                    Train GNN model (default)"
	@echo "  bench-gs                   Benchmark grad SAGEConv kernels"
	@echo "  bench-agg                  Build aggregate kernel benchmark"
	@echo "  dsprep                     Benchmark aggregate kernels"
	@echo "  datasets                   Prepare all datasets for training"
	@echo "  ogbn-arxiv|ogbn-products|ogbn-papers100M  Prepare datasets for training"
	@echo "  all                        Build all targets"
	@echo "  tags                       Generate etags file for project"
	@echo "  clean                      Remove build directory"
	@echo ""
	@echo "Options:"
	@echo "  CC=<compiler>              C compiler                [$(notdir $(CC))]"
	@echo "  CFLAGS=<flags>             Additional compiler flags [$(CFLAGS)]"
	@echo "  LDFLAGS=<flags>            Additional linking flags  [$(LDFLAGS)]"
	@echo "  DEBUG=0|1                  Enable debug build        [$(DEBUG)]"
	@echo "  OPENMP=0|1                 Enable OpenMP             [$(OPENMP)]"
	@echo "  USE_DOUBLE=0|1             Use double precision      [$(USE_DOUBLE)]"
	@echo "  MODE=convergence|benchmark Run mode                  [$(MODE)]"
	@echo "  IMPL=naive|blas|tuned      SAGEConv implementation   [$(IMPL)]"
	@echo "  SPARSE=cs|coo              Sparse graph format       [$(SPARSE)]"
	@echo "  MARCH=<arch>               Target architecture       [$(MARCH)]"
	@echo "  BUILDDIR=<dir>             Build output directory    [$(BUILDDIR)]"
	@echo "  PARTITION=<name>           Config partition          [$(PARTITION)]"
	@echo "  V=0|1                      Verbose output            [$(V)]"
	@echo ""
	@echo "Example: make paragnn DEBUG=1 IMPL=blas SPARSE=coo"
.PHONY: all clean tags help \
        paragnn bench-gs aggregate \
        dsprep arxiv products papers100M

-include $(wildcard $(BUILDDIR)/*.d)
