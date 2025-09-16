CFLAGS = -ggdb -g3 -gdwarf-2 -Wall -Wextra
# CFLAGS += -O3 -march=native
CFLAGS += -O0

all: main

.PHONY: all clean

main: main.o matrix.o layers.o gnn.o arxiv.o simple_graph.o print.o
	gcc $(CFLAGS) $^ -o $@ -lm -lz

main.o: main.c
	gcc $(CFLAGS) -c $< -o $@

matrix.o: matrix.c
	gcc $(CFLAGS) -c $< -o $@

layers.o: layers.c
	gcc $(CFLAGS) -c $< -o $@

gnn.o: gnn.c
	gcc $(CFLAGS) -c $< -o $@

arxiv.o: arxiv.c
	gcc $(CFLAGS) -c $< -o $@

simple_graph.o: simple_graph.c
	gcc $(CFLAGS) -c $< -o $@

print.o: print.c
	gcc $(CFLAGS) -c $< -o $@

arxiv:
	wget http://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip
	unzip arxiv.zip

clean:
	rm -rf main main.o matrix.o simple_graph.o arxiv.o
