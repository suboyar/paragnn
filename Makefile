.DELETE_ON_ERROR:

MAKEFLAGS += -j$(shell nproc)

PARTITION ?= $(or $(SLURM_JOB_PARTITION),default)
-include mkconfigs/$(PARTITION).mk

# Defaulst
CC         ?= gcc
CFLAGS     ?=
LDFLAGS    ?=
DEBUG      ?= 0
OPENMP     ?= 1
USE_DOUBLE ?= 0
IMPL       ?= naive
BUILDDIR   ?= build
TARGET_CPU ?= TARGET_CPU_GENERIC
MARCH      ?= native
DEFS       ?=
V          ?= 0
DATADIR ?= ~/D1/paragnn-ds

# remove trailing slash if included
BUILDDIR := $(patsubst %/,%,$(BUILDDIR))
BENCHDIR = $(BUILDDIR)/benchmark

BASIC_CFLAGS += -D_POSIX_C_SOURCE=200809L \
                -Wfloat-conversion \
                -Werror=implicit-function-declaration \
                -Werror=strict-prototypes \
                -Werror=incompatible-pointer-types
BASIC_CFLAGS += -D$(TARGET_CPU)
BASIC_CFLAGS += $(DEFS)
# Add dummy targets for local header files
BASIC_CFLAGS += -MMD -MP

ifeq ($(DEBUG),1)
    BASIC_CFLAGS += -O0 -ggdb -g3 -gdwarf-2 -march=$(MARCH)
    # suppress ABI warnings from platform-specific vector types
    BASIC_CFLAGS += -Wno-psabi
else
    BASIC_CFLAGS += -O3 -ffast-math -march=$(MARCH) -DNDEBUG
endif

ifeq ($(OPENMP),1)
    BASIC_CFLAGS += -fopenmp
endif

ifeq ($(USE_DOUBLE),1)
    BASIC_CFLAGS += -DUSE_DOUBLE
endif

ifeq ($(IMPL),naive)
    BASIC_CFLAGS += -DSAGECONV_NAIVE_IMPL
else ifeq ($(IMPL),blas)
    BASIC_CFLAGS += -DSAGECONV_BLAS_IMPL
else
    BASIC_CFLAGS += -DSAGECONV_TUNED_IMPL
endif

ALL_CFLAGS = $(strip $(BASIC_CFLAGS) $(CFLAGS))

to_obj = $(patsubst %.c,$(BUILDDIR)/%.o,$1)
to_bench_obj = $(patsubst %.c,$(BENCHDIR)/%.o,$1)

PARAGNN_SRCS = src/main.c src/core.c src/nn.c src/sageconv.c src/matmul_naive.c \
               src/ds.c src/dsinfo.c src/layers.c src/optim.c src/timer.c

GRAD_SAGECONV_SRCS := kernels/grad_sageconv/bench.c \
                      kernels/grad_sageconv/outer_tn/outer_tn_kernel.c \
                      kernels/grad_sageconv/outer_tn/outer_tn_v1.c \
                      kernels/grad_sageconv/outer_tn/outer_tn_v2.c \
                      kernels/grad_sageconv/outer_tn/outer_tn_v3.c \
                      kernels/grad_sageconv/outer_tn/outer_tn_v4.c \
                      kernels/grad_sageconv/outer_tn/outer_tn_v5.c \
                      kernels/grad_sageconv/outer_tn/outer_tn_v6.c \
                      kernels/grad_sageconv/outer_tn/outer_tn_v7.c \
                      kernels/grad_sageconv/grad_mean_aggregate.c \
                      kernels/cache_counter.c \
                      src/core.c \
                      src/ds.c \
                      src/timer.c \
                      src/dsinfo.c \
                      src/layers.c

AGGREGATE_SRCS := kernels/aggregate.c \
                  kernels/cache_counter.c \
                  src/core.c \
                  src/ds.c \
                  src/timer.c \
                  src/dsinfo.c

DSPREP_SRC :=  src/dsprep.c src/dsinfo.c src/core.c

paragnn: $(BUILDDIR)/paragnn
bench-gs: $(BENCHDIR)/bench-gs
# bench-agg: $(BUILDDIR)/bench-agg
dsprep: $(BUILDDIR)/dsprep

all: paragnn bench-gs dsprep

ifeq ($(V),1)
    Q =
    E = @true
else
    Q = @
    E = @echo
endif

$(BUILDDIR)/paragnn: $(call to_obj,$(PARAGNN_SRCS)) | $(BUILDDIR)
	$(E) "  LD    $@"
	$(Q)$(CC) $(ALL_CFLAGS) $(LDFLAGS) -o $@ $^ -lm -lopenblas

$(BENCHDIR)/bench-gs: $(call to_bench_obj,$(GRAD_SAGECONV_SRCS)) | $(BENCHDIR)
	$(E) "  LD    $@"
	$(Q)$(CC) $(ALL_CFLAGS) $(LDFLAGS) -o $@ $^ -lm -lopenblas

$(BUILDDIR)/bench-agg: $(call to_obj,$(AGGREGATE_SRCS)) | $(BUILDDIR)
	$(E) "  LD    $@"
	$(Q)$(CC) $(ALL_CFLAGS) $(LDFLAGS) -o $@ $^ -lm -lopenblas

$(BUILDDIR)/dsprep: $(call to_obj,$(DSPREP_SRC)) | $(BUILDDIR)
	$(E) "  LD    $@"
	$(Q)$(CC) $(BASIC_CFLAGS) -o $@ $^ -lz

$(BUILDDIR)/%.o: %.c
	@mkdir -p $(dir $@)
	$(E) "  CC    $<"
	$(Q)$(CC) $(ALL_CFLAGS) -Isrc/ -c $< -o $@

$(BENCHDIR)/%.o: %.c
	@mkdir -p $(dir $@)
	$(E) "  CC    $<"
	$(Q)$(CC) $(ALL_CFLAGS) -Isrc/ -c $< -o $@

$(BUILDDIR)/grad_sageconv_outer_tn.s: kernels/grad_sageconv_outer_tn.c | $(BUILDDIR)
	$(E) "  ASM   $<"
	$(Q)$(CC) $(ALL_CFLAGS) -DMCA_MARKERS -Isrc/ -S -o $@ $<

arxiv products papers100M: $(BUILDDIR)/dsprep
	./$< -ds $@ -datadir $(DATADIR)

clean:
	rm -rf $(BENCHDIR)
	rm -rf $(BUILDDIR)

help:
	@echo "Usage: make [TARGET] [OPTIONS]"
	@echo ""
	@echo "Targets:"
	@echo "  paragnn                    Train GNN model (default)"
	@echo "  bench-gs                   Benchmark grad SAGEConv kernels"
	@echo "  bench-agg                  Build aggregate kernel benchmark"
	@echo "  dsprep                     Benchmark aggregate kernels"
	@echo "  arxiv|products|papers100M  Prepare datasets for training"
	@echo "  all                        Build all targets"
	@echo "  clean                      Remove build directory"
	@echo ""
	@echo "Options:"
	@echo "  CC=<compiler>              C compiler                [$(notdir $(CC))]"
	@echo "  CFLAGS=<flags>             Additional compiler flags [$(CFLAGS)]"
	@echo "  LDFLAGS=<flags>            Additional linking flags  [$(LDFLAGS)]"
	@echo "  DEBUG=0|1                  Enable debug build        [$(DEBUG)]"
	@echo "  OPENMP=0|1                 Enable OpenMP             [$(OPENMP)]"
	@echo "  USE_DOUBLE=0|1             Use double precision      [$(USE_DOUBLE)]"
	@echo "  IMPL=naive|blas|tuned      SAGEConv implementation   [$(IMPL)]"
	@echo "  MARCH=<arch>               Target architecture       [$(MARCH)]"
	@echo "  BUILDDIR=<dir>             Build output directory    [$(BUILDDIR)]"
	@echo "  PARTITION=<name>           Config partition          [$(PARTITION)]"
	@echo "  V=0|1                      Verbose output            [$(V)]"
	@echo ""
	@echo "Example: make paragnn DEBUG=1 IMPL=blas"

.PHONY: all clean help \
        paragnn bench-gs aggregate \
        dspreprep arxiv products papers100M

-include $(wildcard $(BUILDDIR)/*.d)
