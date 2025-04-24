#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <zlib.h>

#define NOB_IMPLEMENTATION
#define NOB_STRIP_PREFIX
#include "nob.h"

#define ERROR_PRINT(format, ...)                                        \
  fprintf(stderr, "[ERROR] %s:%d - " format "\n", __FILE__, __LINE__, ##__VA_ARGS__)

#define FATAL_ERROR(format, ...)                \
  do {                                          \
    ERROR_PRINT(format, ##__VA_ARGS__);         \
    exit(EXIT_FAILURE);                         \
  } while(0)

#define IDX(i, j, width) ((i) * (width) + (j))
#define IDX3(i, j, k, height, width) ((i) * (height) * (width) + (j) * (width) + (k))

typedef struct {
  size_t num_edges;
  size_t num_nodes;
  size_t num_node_features;
  size_t num_label_classes;
  uint32_t *node_year;
  uint32_t *edge_index; // Graph connectivity in COO format with shape [2, num_edges]
  double   *x; // Node feature matrix with shape [num_nodes, num_node_features] (features)
  uint32_t *y; // node-level targets of shape [num_nodes, num_label_classes] or graph-level targets of shape [1, num_label_classes] (labels)
} ogb_arxiv_t;

#define CHUNK 0x1000 // 4kb window size

String_Builder sb = {0};

String_View read_gz(char* file_path)
{
  gzFile file = gzopen(file_path, "rb");
  if (!file) { FATAL_ERROR("gzopen of '%s' failed: %s", file_path, strerror(errno)); }


  sb.count = 0;
  char buffer[CHUNK];
  while (1) {
    int bytes_read = gzread(file, buffer, CHUNK);

    if (bytes_read <= 0) {
      break;
    }

    sb_append_buf(&sb, buffer, bytes_read);
  }

  int err = 0;
  const char * error_string;
  if (!gzeof(file)) {
    error_string = gzerror (file, &err);
  }

  gzclose(file);

  if (err){
    FATAL_ERROR("gzread of '%s' failed: %s", file_path, error_string);
  }

  sb_append_null(&sb);
  return sb_to_sv(sb);
}

#define RAW_PATH "/home/sboyar/D1/dataset/ogb/data/ogbn_arxiv/raw"

void load_data(ogb_arxiv_t *arxiv)
{
  String_View sv = {0};

  sv = read_gz(RAW_PATH"/num-node-list.csv.gz");
  String_View num_node_sv = sv_chop_by_delim(&sv, '\n');
  const char *num_node_cstr = temp_sv_to_cstr(num_node_sv);
  arxiv->num_nodes = atol(num_node_cstr);

  sv = read_gz(RAW_PATH"/num-edge-list.csv.gz");
  String_View num_edge_sv = sv_chop_by_delim(&sv, '\n');
  const char *num_edge_cstr = temp_sv_to_cstr(num_edge_sv);
  arxiv->num_edges = atol(num_edge_cstr);

  sv = read_gz(RAW_PATH"/node_year.csv.gz");
  arxiv->node_year = malloc(arxiv->num_nodes * sizeof(*arxiv->node_year));
  if (!arxiv->node_year) { FATAL_ERROR("Failed to allocate memory for node years"); }
  for (size_t i = 0; sv.count > 0; i++) {
    temp_reset();
    String_View year_sv = sv_chop_by_delim(&sv, '\n');
    // Jump over empty lines if any
    if (year_sv.data[year_sv.count-1] == '\0') { continue; }

    const char *year_cstr = temp_sv_to_cstr(year_sv);
    assert(strcmp(year_cstr, "") != 0);
    arxiv->node_year[i] = atol(year_cstr);
    assert (i < arxiv->num_nodes);
  }

  // TODO: Programatically find the number of labels through counting labels in
  // the first line of ogbn_arxiv/raw/node-label.csv.gz
  arxiv->num_label_classes = 1;
  arxiv->y = malloc(arxiv->num_nodes * arxiv->num_label_classes * sizeof(*arxiv->y));
  if (!arxiv->y) { FATAL_ERROR("Failed to allocate memory for labels"); }
  sv = read_gz(RAW_PATH"/node-label.csv.gz");

  for (size_t i = 0; sv.count > 0; i++) {
    String_View labels_sv = sv_chop_by_delim(&sv, '\n');
    // Jump over empty lines if any
    if (labels_sv.data[labels_sv.count-1] == '\0') { continue; }

    for (size_t j = 0; labels_sv.count > 0; j++) {
      temp_reset();
      String_View label_sv = sv_chop_by_delim(&labels_sv, ',');
      const char *label_cstr = temp_sv_to_cstr(label_sv);
      assert(strcmp(label_cstr, "") != 0);
      arxiv->y[IDX(i, j, arxiv->num_label_classes)] = atol(label_cstr);
      assert (j < arxiv->num_label_classes);
    }
    assert (i < arxiv->num_nodes);
  }

  // TODO: Programatically find the number of features through counting features
  // in the first line of ogbn_arxiv/raw/node-feat.csv.gz
  arxiv->num_node_features = 128;
  arxiv->x = malloc(arxiv->num_nodes * arxiv->num_node_features * sizeof(*arxiv->x));
  if (!arxiv->x) { FATAL_ERROR("Failed to allocate memory for features"); }
  sv = read_gz(RAW_PATH"/node-feat.csv.gz");

  for (size_t i = 0; sv.count > 0; i++) {
    String_View feats_sv = sv_chop_by_delim(&sv, '\n');
    // Jump over empty lines if any
    if (feats_sv.data[feats_sv.count-1] == '\0') { continue; }

    for (size_t j = 0; feats_sv.count > 0; j++) {
      temp_reset();
      String_View feat_sv = sv_chop_by_delim(&feats_sv, ',');
      const char *feat_cstr = temp_sv_to_cstr(feat_sv);
      assert(strcmp(feat_cstr, "") != 0);
      arxiv->x[IDX(i, j, arxiv->num_node_features)] = atof(feat_cstr);
      assert(j < arxiv->num_node_features);
    }
    assert(i < arxiv->num_nodes);
  }

  arxiv->edge_index = malloc(2 * arxiv->num_edges * sizeof(*arxiv->edge_index));
  if (!arxiv->edge_index) { FATAL_ERROR("Failed to allocate memory for edges"); }
  sv = read_gz(RAW_PATH"/edge.csv.gz");

  for (size_t i = 0; sv.count > 0; i++) {
    String_View edges_sv = sv_chop_by_delim(&sv, '\n');
    // Jump over empty lines if any
    if (edges_sv.data[edges_sv.count-1] == '\0') { continue; }

    for (size_t j = 0; edges_sv.count > 0; j++) {
      temp_reset();
      String_View edge_sv = sv_chop_by_delim(&edges_sv, ',');
      const char *edge_cstr = temp_sv_to_cstr(edge_sv);
      assert(strcmp(edge_cstr, "") != 0);
      arxiv->edge_index[IDX(j, i, arxiv->num_edges)] = atol(edge_cstr);

      assert(j < 2);
    }
    assert(i < arxiv->num_edges);
  }

}

int main(void)
{
  ogb_arxiv_t arxiv = {0};

  load_data(&arxiv);

  free(arxiv.node_year);
  free(arxiv.edge_index);
  free(arxiv.x);
  free(arxiv.y);
  sb_free(sb);
  return 0;
}

