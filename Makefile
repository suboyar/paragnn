.DELETE_ON_ERROR:

PARTITION ?= default
-include mkconfigs/$(PARTITION).mk

# Defaults
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
BUILDDIR   := $(patsubst %/,%,$(BUILDDIR))

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
    BASIC_CFLAGS += -O0 -ggdb -g3 -gdwarf-2
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

to_obj = $(addprefix $(BUILDDIR)/,$(notdir $(patsubst %.c,%.o,$1)))

PARAGNN_SRCS = src/main.c src/core.c src/nn.c src/sageconv.c src/matmul_naive.c \
               src/ds.c src/dsinfo.c src/layers.c src/optim.c src/timer.c

GRAD_SAGECONV_SRCS := kernels/grad_sageconv_common.c \
                      kernels/grad_sageconv_fused.c \
                      kernels/grad_sageconv_gemm_tn.c \
                      kernels/grad_sageconv_outer_tn.c \
                      kernels/grad_sageconv.c \
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
bench-gs: $(BUILDDIR)/bench-gs
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

$(BUILDDIR)/bench-gs: $(call to_obj,$(GRAD_SAGECONV_SRCS)) | $(BUILDDIR)
	$(E) "  LD    $@"
	$(Q)$(CC) $(ALL_CFLAGS) $(LDFLAGS) -o $@ $^ -lm -lopenblas

$(BUILDDIR)/bench-agg: $(call to_obj,$(AGGREGATE_SRCS)) | $(BUILDDIR)
	$(E) "  LD    $@"
	$(Q)$(CC) $(ALL_CFLAGS) $(LDFLAGS) -o $@ $^ -lm -lopenblas

$(BUILDDIR)/dsprep: $(call to_obj,$(DSPREP_SRC)) | $(BUILDDIR)
	$(E) "  LD    $@"
	$(Q)$(CC) $(BASIC_CFLAGS) -o $@ $^ -lz

$(BUILDDIR)/%.o: src/%.c | $(BUILDDIR)
	$(E) "  CC    $<"
	$(Q)$(CC) $(ALL_CFLAGS) -c $< -o $@

$(BUILDDIR)/%.o: kernels/%.c | $(BUILDDIR)
	$(E) "  CC    $<"
	$(Q)$(CC) $(ALL_CFLAGS) -Isrc/ -c $< -o $@

$(BUILDDIR):
	mkdir -p $@

arxiv products papers100M: $(BUILDDIR)/dsprep
	./$< -ds $@ -datadir $(DATADIR)

clean:
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
