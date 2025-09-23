# CFLAGS = -O3 -march=native -DNDEBUG
CFLAGS = -O0 -ggdb -g3 -gdwarf-2
CFLAGS += -std=c17 -D_POSIX_C_SOURCE=200809L
CFLAGS += -fopenmp
CFLAGS += -Wall -Wextra
CFLAGS += -Werror=implicit-function-declaration -Werror=incompatible-pointer-types
CLIBS = -lm -lz

all: main

main: main.o matrix.o layers.o gnn.o graph.o arxiv.o simple_graph.o
	gcc $(CFLAGS) $^ -o $@ $(CLIBS)

%.o: %.c
	gcc $(CFLAGS) -c $< -o $@

gen-asm: asm/main.s asm/matrix.s asm/layers.s asm/gnn.s asm/arxiv.s asm/simple_graph.s

asm/%.s: %.c | asm
	gcc $(CFLAGS) -fverbose-asm -fno-omit-frame-pointer -masm=intel -S $< -o $@

asm:
	mkdir -p $@

arxiv: arxiv.zip
	unzip arxiv.zip

arxiv.zip:
	wget http://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip

clean:
	rm -rf main *.o asm/

.PHONY: all clean gen-asm
