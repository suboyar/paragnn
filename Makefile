# Debug flags (default)
CFLAGS = -O0 -ggdb -g3 -gdwarf-2
# Release flags
CFLAGS_RELEASE = -O3 -march=native -DNDEBUG -DUSE_OGB_ARXIV

# Common flags
CFLAGS += -std=c17 -D_POSIX_C_SOURCE=200809L
CFLAGS += -fopenmp
CFLAGS += -Wall -Wextra -Wfloat-conversion
CFLAGS += -Werror=implicit-function-declaration -Werror=incompatible-pointer-types

CFLAGS_RELEASE += -std=c17 -D_POSIX_C_SOURCE=200809L
CFLAGS_RELEASE += -fopenmp
CFLAGS_RELEASE += -Wall -Wextra -Wfloat-conversion
CFLAGS_RELEASE += -Werror=implicit-function-declaration -Werror=incompatible-pointer-types

CLIBS = -lm

CFLAGS_OGB = -Wall -Wextra -std=c17 -D_POSIX_C_SOURCE=200809L
CFLAGS_OGB += -O3
CLIBS_OGB = -lz


all: main ogb

release: clean
	$(MAKE) CFLAGS="$(CFLAGS_RELEASE)" main

main: main.o matrix.o layers.o gnn.o graph.o benchmark.o
	gcc $(CFLAGS) $^ -o $@ $(CLIBS)

ogb: ogb.o				# Decompreses the dataset
	gcc $(CFLAGS_OGB) $^ -o $@ $(CLIBS_OGB)

%.o: %.c
	gcc $(CFLAGS) -c $< -o $@

gen-asm: asm/main.s asm/matrix.s asm/layers.s asm/gnn.s asm/arxiv.s asm/simple_graph.s

asm/%.s: %.c | asm
	gcc $(CFLAGS) -fverbose-asm -fno-omit-frame-pointer -masm=intel -S $< -o $@

asm:
	mkdir -p $@

dataset/arxiv: arxiv.zip
	unzip arxiv.zip -d dataset/

arxiv.zip:
	wget http://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip

clean:
	rm -rf main *.o asm/

.PHONY: all clean gen-asm release
