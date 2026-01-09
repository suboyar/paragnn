#define _GNU_SOURCE
#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <strings.h>
#include <inttypes.h>
#include <sys/mman.h>

#include <omp.h>

#include "dataset.h"

static const uint32_t num_inputs = 169343;
static const uint32_t num_features = 128;
static const uint32_t num_classes = 40;

typedef enum {
    TRAIN_PARTITION,
    VALID_PARTITION,
    TEST_PARTITION,
    INVALID_PARTITION
} Partition;

typedef struct {
    Partition split;
    size_t idx;
} NodeMapping;

static inline size_t range_len(Range r) { return r.end - r.start; }

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

static void build_node_mapping(NodeMapping* map,  uint32_t num_inputs,
                               uint32_t* train_idx, uint32_t train_size,
                               uint32_t* valid_idx, uint32_t valid_size,
                               uint32_t* test_idx, uint32_t test_size)
{
    // Initialize all as invalid
    for (size_t i = 0; i < num_inputs; i++)
        map[i] = (NodeMapping){INVALID_PARTITION, SIZE_MAX};

    for (size_t i = 0; i < train_size; i++)
        map[train_idx[i]] = (NodeMapping){TRAIN_PARTITION, i};
    for (size_t i = 0; i < valid_size; i++)
        map[valid_idx[i]] = (NodeMapping){VALID_PARTITION, i};
    for (size_t i = 0; i < test_size; i++)
        map[test_idx[i]] = (NodeMapping){TEST_PARTITION, i};
}

static inline void gather_features(double *dest, double *src, uint32_t *split_idx, size_t split_size)
{
#pragma omp parallel for
    for (size_t i = 0; i < split_size; i++) {
        size_t orig = split_idx[i];
        memcpy(&dest[i * num_features], &src[orig * num_features],
               num_features * sizeof(double));
    }

}

static void load_inputs(double *dest,
                        uint32_t *train_idx, Range train_range,
                        uint32_t *valid_idx, Range valid_range,
                        uint32_t *test_idx, Range test_range)
{
    double* temp = malloc(num_inputs * num_features * sizeof(double));

    const char *file = "./arxiv/processed/node-feat.csv";
    int fd = open(file, O_RDONLY);
    if (fd < 0) ERROR("Could not open %s: %s", file, strerror(errno));
    struct stat sb;
    fstat(fd, &sb);
    char* data = mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);

    char** line_starts = malloc((num_inputs + 1) * sizeof(char*));
    line_starts[0] = data;
    size_t line = 1;
    for (char* p = data; p < data + sb.st_size; p++) {
        if (*p == '\n' && line < num_inputs) {
            line_starts[line++] = p + 1;
        }
    }

#pragma omp parallel for
    for (size_t i = 0; i < num_inputs; i++) {
        char* p = line_starts[i];
        for (size_t j = 0; j < num_features; j++) {
            temp[i*num_features + j] = parse_double(&p);
            if (*p == ',') p++;
        }
    }

    double *train = dest + train_range.start * num_features;
    double *valid = dest + valid_range.start * num_features;
    double *test = dest + test_range.start * num_features;
    gather_features(train, temp, train_idx, range_len(train_range));
    gather_features(valid, temp, valid_idx, range_len(valid_range));
    gather_features(test, temp, test_idx, range_len(test_range));

    free(temp);
    free(line_starts);
    munmap(data, sb.st_size);
    close(fd);

}

static void load_labels(uint32_t *dest,
                        uint32_t *train_idx, Range train_range,
                        uint32_t *valid_idx, Range valid_range,
                        uint32_t *test_idx, Range test_range)
{
    size_t* temp = malloc(num_inputs * sizeof(size_t));

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
        temp[i++] = (size_t)parse_u32(&p);
        if (*p == '\n') p++;
    }

    uint32_t *train = dest + train_range.start;
    for (size_t i = 0; i < range_len(train_range); i++) {
        train[i] = temp[train_idx[i]];
    }

    uint32_t *valid = dest + valid_range.start;
    for (size_t i = 0; i < range_len(valid_range); i++) {
        valid[i] = temp[valid_idx[i]];
    }

    uint32_t *test = dest + test_range.start;
    for (size_t i = 0; i < range_len(test_range); i++) {
        test[i] = temp[test_idx[i]];
    }

    munmap(data, sb.st_size);
    close(fd);
    free(temp);
}

typedef struct {
    uint32_t *train_edges;
    uint32_t *valid_edges;
    uint32_t *test_edges;
    uint32_t train_count;
    uint32_t valid_count;
    uint32_t test_count;
    uint32_t raw_count;
} EdgeTemp;

static EdgeTemp* count_edges(NodeMapping* map)
{
    const char *file = "./arxiv/processed/edge.csv";
    int fd = open(file, O_RDONLY);
    if (fd < 0) ERROR("Could not open %s: %s", file, strerror(errno));
    struct stat sb;
    fstat(fd, &sb);
    char *data = mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);

    size_t raw_edges = 0;
    for (char *p = data; p < data + sb.st_size; p++) {
        if (*p == '\n') raw_edges++;
    }

    uint32_t *train_edges = malloc(2 * raw_edges * sizeof(*train_edges));
    uint32_t *valid_edges = malloc(2 * raw_edges * sizeof(*valid_edges));
    uint32_t *test_edges  = malloc(2 * raw_edges * sizeof(*test_edges));
    uint32_t train_count = 0, valid_count = 0, test_count = 0;

    char *p = data, *end = data + sb.st_size;
    while (p < end) {
        uint32_t src = parse_u32(&p);
        if (*p == ',') p++;
        uint32_t dst = parse_u32(&p);
        if (*p == '\n') p++;

        NodeMapping s = map[src], d = map[dst];
        if (s.split == d.split && s.split != INVALID_PARTITION) {
            uint32_t *e, *count;
            switch (s.split) {
            case TRAIN_PARTITION: e = train_edges; count = &train_count; break;
            case VALID_PARTITION: e = valid_edges; count = &valid_count; break;
            case TEST_PARTITION:  e = test_edges;  count = &test_count;  break;
            default: continue;
            }

            e[2 * (*count) + 0] = s.idx;
            e[2 * (*count) + 1] = d.idx;
            (*count)++;
        }
    }

    munmap(data, sb.st_size);
    close(fd);

    EdgeTemp *temp = malloc(sizeof(*temp));
    *temp = (EdgeTemp) {
        .train_edges = train_edges,
        .valid_edges = valid_edges,
        .test_edges  = test_edges,
        .train_count = train_count,
        .valid_count = valid_count,
        .test_count  = test_count,
        .raw_count   = raw_edges
    };

    return temp;
}

static void load_edges(uint32_t *dest, EdgeTemp *temp,
                       Range train_range, Range valid_range, Range test_range)
{
    memcpy(dest + 2*train_range.start, temp->train_edges, 2*range_len(train_range) * sizeof(uint32_t));
    memcpy(dest + 2*valid_range.start, temp->valid_edges, 2*range_len(valid_range) * sizeof(uint32_t));
    memcpy(dest + 2*test_range.start, temp->test_edges, 2*range_len(test_range) * sizeof(uint32_t));

    free(temp->train_edges);
    free(temp->valid_edges);
    free(temp->test_edges);
    free(temp);
}

Dataset* load_arxiv_dataset()
{
    double t = omp_get_wtime();

    uint32_t *train_idx, *valid_idx, *test_idx;
    uint32_t train_size = load_split("./arxiv/processed/train.csv", &train_idx);
    uint32_t valid_size = load_split("./arxiv/processed/valid.csv", &valid_idx);
    uint32_t test_size = load_split("./arxiv/processed/test.csv", &test_idx);

#ifndef NDEBUG
    train_size = 700, valid_size = 400, test_size = 600;
#endif

    NodeMapping* node_map = malloc(num_inputs * sizeof(*node_map));
    build_node_mapping(node_map, num_inputs,
                       train_idx, train_size, valid_idx, valid_size, test_idx,  test_size);

    Dataset *data = malloc(sizeof(*data));

    // Count Edge
    EdgeTemp *edge_temp = count_edges(node_map);
    uint32_t train_edges = edge_temp->train_count;
    uint32_t valid_edges = edge_temp->valid_count;
    uint32_t test_edges  = edge_temp->test_count;
    uint32_t raw_edges   = edge_temp->raw_count;
    uint32_t num_edges = train_edges + valid_edges + test_edges;

    // Add size info
#ifndef NDEBUG
    data->num_inputs = train_size + valid_size + test_size;
#else
    data->num_inputs = num_inputs;
#endif
    data->num_edges = num_edges;
    data->num_features = num_features;
    data->num_classes = num_classes;

    data->train = (Slice) {
        .node = { .start=0, .end=train_size },
        .edge = { .start=0, .end=train_edges }
    };

    data->valid = (Slice) {
        .node = { .start=data->train.node.end, .end=(data->train.node.end + valid_size) },
        .edge = { .start=data->train.edge.end, .end=(data->train.edge.end + valid_edges) }
    };

    data->test = (Slice) {
        .node = { .start=data->valid.node.end, .end=(data->valid.node.end + test_size) },
        .edge = { .start=data->valid.edge.end, .end=(data->valid.edge.end + test_edges) }
    };

    data->full = (Slice) {
        .node = { .start=data->train.node.start, .end=data->test.node.end },
        .edge = { .start=data->train.edge.start, .end=data->test.edge.end }
    };

    // Features
    data->inputs = malloc(num_inputs * num_features * sizeof(*data->inputs));
    load_inputs(data->inputs,
                train_idx, data->train.node,
                valid_idx, data->valid.node,
                test_idx, data->test.node);

    // Labels
    data->labels = malloc(num_inputs * sizeof(*data->labels));
    load_labels(data->labels,
                train_idx, data->train.node,
                valid_idx, data->valid.node,
                test_idx, data->test.node);

    // Edges
    data->edges.data = malloc(2 * num_edges * sizeof(*data->edges.data));
    load_edges(data->edges.data, edge_temp,
               data->train.edge,
               data->valid.edge,
               data->test.edge);

    free(train_idx);
    free(valid_idx);
    free(test_idx);
    free(node_map);

    printf("Loaded OGB-ARXIV [%u/%u/%u] in %.2fs (edges: %u -> %u, -%.0f%%)\n",
           train_size, valid_size, test_size,
           omp_get_wtime() - t,
           raw_edges, train_edges + valid_edges + test_edges,
           100 - 100.0 * (train_edges + valid_edges + test_edges) / raw_edges);

    return data;
}

void destroy_dataset(Dataset *ds)
{
    free(ds->edges.data);
    free(ds->inputs);
    free(ds->labels);
    free(ds);
}
