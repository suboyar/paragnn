#define _GNU_SOURCE
#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <strings.h>
#include <inttypes.h>
#include <sys/mman.h>

#include <omp.h>

#include "dataset.h"

static const uint32_t num_nodes = 169343;
static const uint32_t num_features = 128;
static const uint32_t num_classes = 40;

static const char *split_name[] = {
    [SPLIT_TRAIN] = "train",
    [SPLIT_VALID] = "valid",
    [SPLIT_TEST]  = "test",
};

// This is specifically design to only conver cases for node-feat.csv with no space handling
static inline double parse_double(char** pp)
{
    char* p = *pp;
    double sign = 1.0;

    if (*p == '-') { sign = -1.0; p++; }
    else if (*p == '+') { p++; }

    int64_t intpart = 0;
    while (*p >= '0' && *p <= '9') {
        intpart = intpart * 10 + (*p++ - '0');
    }

    double val = (double)intpart;

    if (*p == '.') {
        p++;
        double scale = 0.1;
        while (*p >= '0' && *p <= '9') {
            val += (*p++ - '0') * scale;
            scale *= 0.1;
        }
    }

    if (*p == 'e' || *p == 'E') {
        p++;
        int exp_sign = 1;
        if (*p == '-') { exp_sign = -1; p++; }
        else if (*p == '+') { p++; }

        int exp = 0;
        while (*p >= '0' && *p <= '9') {
            exp = exp * 10 + (*p++ - '0');
        }

        if (exp_sign > 0) {
            while (exp-- > 0) val *= 10.0;
        } else {
            while (exp-- > 0) val *= 0.1;
        }
    }

    *pp = p;
    return sign * val;
}

// This is specifically design to only be used for node indicies that aren't bigger
// then 32bit value. It does not do any space checking or clean-up.
static inline uint32_t parse_u32(char** pp)
{
    char* p = *pp;

    uint32_t val = 0;
    while (*p >= '0' && *p <= '9') {
        val = val * 10 + (*p++ - '0');
    }

    *pp = p;
    return val;
}

static uint32_t load_split(const char *path, uint32_t **split)
{
    int fd = open(path, O_RDONLY);
    if (fd < 0) ERROR("Could not open %s: %s", path, strerror(errno));
    struct stat sb;
    fstat(fd, &sb);
    char* data = mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);

    char* p = data;
    char* end = data + sb.st_size;
    uint32_t size = 0;
    while (p < end) {
        if (*p == '\n') size++;
        p++;
    }

    *split = malloc(size * sizeof(**split));

    p = data;
    size_t count = 0;
    while (p < end) {
        (*split)[count++] = parse_u32(&p);
        if (*p == '\n') p++;
    }

    close(fd);
    return size;
}

static void build_node_mapping(uint32_t* map,  uint32_t num_nodes, uint32_t* split_idx, uint32_t split_size)
{
    if (num_nodes >= UINT32_MAX)
    {
        ERROR("num_nodes too large for UINT32_MAX sentinel");
    }

    memset(map, 0xFF, num_nodes * sizeof(*map));

    for (uint32_t i = 0; i < split_size; i++)
    {
        map[split_idx[i]] = i;
    }
}

static void load_inputs(double *dest, uint32_t num_nodes)
{
    const char *file = "./arxiv/processed/node-feat.csv";
    int fd = open(file, O_RDONLY);
    if (fd < 0) ERROR("Could not open %s: %s", file, strerror(errno));
    struct stat sb;
    fstat(fd, &sb);
    char* data = mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);

    char** line_starts = malloc((num_nodes + 1) * sizeof(char*));
    line_starts[0] = data;
    size_t line = 1;
    for (char* p = data; p < data + sb.st_size; p++) {
        if (*p == '\n' && line < num_nodes) {
            line_starts[line++] = p + 1;
        }
    }

#pragma omp parallel for
    for (size_t i = 0; i < num_nodes; i++) {
        char* p = line_starts[i];
        for (size_t j = 0; j < num_features; j++) {
            dest[i*num_features + j] = parse_double(&p);
            if (*p == ',') p++;
        }
    }

    free(line_starts);
    munmap(data, sb.st_size);
    close(fd);

}

static void load_labels(uint32_t *dest)
{
    const char *file = "./arxiv/processed/node-label.csv";
    int fd = open(file, O_RDONLY);
    if (fd < 0) ERROR("Could not open %s: %s", file, strerror(errno));
    struct stat sb;
    fstat(fd, &sb);
    char* data = mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);

    char* p = data;
    char* end = data + sb.st_size;

    size_t i = 0;
    while (p < end) {
        dest[i++] = parse_u32(&p);
        if (*p == '\n') p++;
    }

    munmap(data, sb.st_size);
    close(fd);
}

typedef struct {
    uint32_t *data;   // pairs: [src0, dst0, src1, dst1, ...]
    uint32_t count;
} RawEdges;

static RawEdges load_edges(void)
{
    const char *file = "./arxiv/processed/edge.csv";
    int fd = open(file, O_RDONLY);
    if (fd < 0) ERROR("Could not open %s: %s", file, strerror(errno));
    struct stat sb;
    fstat(fd, &sb);
    char *data = mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);

    uint32_t count = 0;
    for (char *p = data; p < data + sb.st_size; p++) {
        if (*p == '\n') count++;
    }

    uint32_t *edges = malloc(2 * count * sizeof(*edges));

    char *p = data, *end = data + sb.st_size;
    size_t i = 0;
    while (p < end) {
        edges[2*i]     = parse_u32(&p);
        if (*p == ',') p++;
        edges[2*i + 1] = parse_u32(&p);
        if (*p == '\n') p++;
        i++;
    }

    munmap(data, sb.st_size);
    close(fd);

    return (RawEdges){ .data = edges, .count = count };
}

Dataset* load_arxiv_dataset(void)
{
    double t = omp_get_wtime();

    Dataset *ds = malloc(sizeof(*ds));

    // Edges
    double t_e = omp_get_wtime();
    RawEdges raw_edges = load_edges();
    ds->edges.data = raw_edges.data;
    t_e = omp_get_wtime() - t_e;

    // Features
    double t_f = omp_get_wtime();
    ds->nodes = malloc(num_nodes * num_features * sizeof(*ds->nodes));
    ds->num_nodes = num_nodes;
    load_inputs(ds->nodes, num_nodes);
    t_f = omp_get_wtime() - t_f;

    // Labels
    double t_l = omp_get_wtime();
    ds->labels = malloc(num_nodes * sizeof(*ds->labels));
    load_labels(ds->labels);
    t_l = omp_get_wtime() - t_l;

    // Add size info
    ds->num_edges = raw_edges.count;
    ds->num_features = num_features;
    ds->num_classes = num_classes;

    printf("Loaded OGB-ARXIV in %.2fs\n", omp_get_wtime() - t);
    printf("    loading edges: %.2fs\n", t_e);
    printf("    loading features: %.2fs\n", t_f);
    printf("    loading labels: %.2fs\n", t_l);
    return ds;
}

#define INVALID_IDX UINT32_MAX

Dataset *split_dataset(Dataset *src, Split split)
{
    double t = omp_get_wtime();
    uint32_t *split_idx, split_size;

    static const char *split_paths[] = {
        [SPLIT_TRAIN] = "./arxiv/processed/train.csv",
        [SPLIT_VALID] = "./arxiv/processed/valid.csv",
        [SPLIT_TEST]  = "./arxiv/processed/test.csv",
    };

    if (split < 0 || split > SPLIT_TEST)
    {
        ERROR("Unknown split: %d", split);
    }

    split_size = load_split(split_paths[split], &split_idx);

#if defined(REDUCED_GRAPH)
    static const uint32_t reduced_sizes[] = {
        [SPLIT_TRAIN] = 700,
        [SPLIT_VALID] = 400,
        [SPLIT_TEST]  = 600,
    };
    split_size = reduced_sizes[split];
#endif

    Dataset *ds = malloc(sizeof(*ds));
    ds->num_features = num_features;
    ds->num_classes  = num_classes;

    // Gather features
    ds->nodes = malloc(split_size * num_features * sizeof(*ds->nodes));
#pragma omp parallel for
    for (size_t i = 0; i < split_size; i++) {
        memcpy(&ds->nodes[i * num_features],
               &src->nodes[split_idx[i] * num_features],
               num_features * sizeof(*ds->nodes));
    }

    // Gather labels
    ds->labels = malloc(split_size * sizeof(*ds->labels));
    for (size_t i = 0; i < split_size; i++) {
        ds->labels[i] = src->labels[split_idx[i]];
    }

    uint32_t *node_map = malloc(num_nodes * sizeof(*node_map));
    build_node_mapping(node_map, num_nodes, split_idx, split_size);

    // Filter and remap edges
    ds->edges.data = malloc(2 * src->num_edges * sizeof(*ds->edges.data)); // Worst case: every edge survives
    uint32_t edge_count = 0;
    for (uint32_t i = 0; i < src->num_edges; i++) {
        uint32_t s = node_map[src->edges.data[2*i]];
        uint32_t d = node_map[src->edges.data[2*i + 1]];
        if (s != INVALID_IDX && d != INVALID_IDX) {
            ds->edges.data[2*edge_count]     = s;
            ds->edges.data[2*edge_count + 1] = d;
            edge_count++;
        }
    }
    // Shrink to actual size
    ds->edges.data = realloc(ds->edges.data, 2 * edge_count * sizeof(*ds->edges.data));

    ds->num_nodes = split_size;
    ds->num_edges  = edge_count;

    free(node_map);
    free(split_idx);


    printf("Split OGB-ARXIV [%s] in %.2fs\n", split_name[split], omp_get_wtime() - t);
    return ds;
}

void free_dataset(Dataset *ds)
{
    free(ds->edges.data);
    free(ds->nodes);
    free(ds->labels);
    free(ds);
}
