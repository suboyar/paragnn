# CFLAGS = -O3 -march=native
CFLAGS = -O0 -ggdb -g3 -gdwarf-2
CFLAGS += -std=c17 -D_POSIX_C_SOURCE=200809L
CFLAGS += -Wall -Wextra -Werror=implicit-function-declaration
CFLAGS += -Werror=implicit-function-declaration -Werror=incompatible-pointer-types
CFLAGS += -DNEWWAY -DCOL_MAJOR

all: main

main: main.o matrix.o layers.o gnn.o arxiv.o simple_graph.o
	gcc $(CFLAGS) $^ -o $@ -lm -lz

%.o: %.c
	gcc $(CFLAGS) -c $< -o $@

# main.o: main.c
# 	gcc $(CFLAGS) -c $< -o $@

# matrix.o: matrix.c
# 	gcc $(CFLAGS) -c $< -o $@

# layers.o: layers.c
# 	gcc $(CFLAGS) -c $< -o $@

# gnn.o: gnn.c
# 	gcc $(CFLAGS) -c $< -o $@

# arxiv.o: arxiv.c
# 	gcc $(CFLAGS) -c $< -o $@

# simple_graph.o: simple_graph.c
# 	gcc $(CFLAGS) -c $< -o $@

gen-asm: asm/main.s asm/matrix.s asm/layers.s asm/gnn.s asm/arxiv.s asm/simple_graph.s

asm/%.s: %.c | asm
	gcc $(CFLAGS) -fverbose-asm -fno-omit-frame-pointer -masm=intel -S $< -o $@

asm:
	mkdir -p $@

arxiv:
	wget http://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip
	unzip arxiv.zip

clean:
	rm -rf main *.o asm/

.PHONY: all clean gen-asm
