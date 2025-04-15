#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <zlib.h>

#define NOB_IMPLEMENTATION
#include "nob.h"

#define ERROR_PRINT(format, ...)                                        \
  fprintf(stderr, "[ERROR] %s:%d - " format "\n", __FILE__, __LINE__, ##__VA_ARGS__)

#define FATAL_ERROR(format, ...)                \
  do {                                          \
    ERROR_PRINT(format, ##__VA_ARGS__);         \
    exit(EXIT_FAILURE);                         \
  } while(0)


typedef struct {
    uint64_t *items;            // need to flaot32 for x
    size_t count;
    size_t capacity;
} da_t;

typedef struct {
  uint32_t num_nodes;
  uint32_t num_node_features;
  da_t *node_year;
  da_t edge_index;          // Graph connectivity in COO format with shape [2, num_edges]
  da_t *x;                       // Node feature matrix with shape [num_nodes, num_node_features] (features)
  da_t *y;                       // node-level targets of shape [num_nodes, *] or graph-level targets of shape [1, *] (labels)

  // void edge_attr;               // Edge feature matrix with shape [num_edges, num_edge_features] (Not relevant in ogb-arxiv)
  // void pos;                     // Node position matrix with shape [num_nodes, num_dimensions] (Not relevant in ogb-arxiv)  

  long *items;
  size_t count;
  size_t capacity;
} data_t;

#define CHUNK 0x1000 // 4kb window size

// TODO: The read_gz function should return the entire string, which should then
//       be parsed as either uint64 or float32 as necessary.

void read_gz(char* file_path, da_t *da)
{
  gzFile file = gzopen(file_path, "rb");
  if (!file) { FATAL_ERROR("gzopen of '%s' failed: %s", file_path, strerror(errno)); }

  Nob_String_Builder sb = {0};
  char buffer[CHUNK];
  int bytes_read;
  do {
    bytes_read = gzread(file, buffer, CHUNK);
    // printf("bytes_read: %d\n", bytes_read);
    if (bytes_read > 0) {
      for (int i = 0; i < bytes_read; i++) {
        if (buffer[i] != '\n') {
          nob_da_append(&sb, buffer[i]);
        } else {
          nob_da_append(&sb, '\0');
          nob_da_append(da, atoll(sb.items));
          sb.count = 0;
        }
      }
    }
  } while (bytes_read > 0);

  if (!gzeof(file)) {
    const char * error_string;
    int err = 0;
    error_string = gzerror (file, &err);
    if (err) { FATAL_ERROR("gzread error: %s", error_string); }
  }

  gzclose(file);

}

int main(void)
{
  char* raw_path = "/home/sboyar/D1/dataset/ogb/data/ogbn_arxiv/raw";
  char label_path[1024];
  strcpy(label_path, raw_path);
  strcat(label_path, "/node-label.csv.gz");
  
  da_t labels = {0};z
  read_gz(label_path, &labels);
  nob_da_foreach(uint64_t, label, &labels) {
    printf("%lu\n", *label);
  }

  return 0;
}

