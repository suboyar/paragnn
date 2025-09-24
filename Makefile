# CFLAGS = -O3 -march=native -DNDEBUG
CFLAGS = -O0 -ggdb -g3 -gdwarf-2
CFLAGS += -std=c17 -D_POSIX_C_SOURCE=200809L
CFLAGS += -fopenmp
CFLAGS += -Wall -Wextra
CFLAGS += -Werror=implicit-function-declaration -Werror=incompatible-pointer-types
CLIBS = -lm

CFLAGS_OGB = -Wall -Wextra -std=c17 -D_POSIX_C_SOURCE=200809L
# CFLAGS_OGB += -O0 -ggdb
CFLAGS_OGB += -O3 -ggdb
CLIBS_OGB = -lz

all: main ogb

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

.PHONY: all clean gen-asm
