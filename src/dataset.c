#define _GNU_SOURCE
#include <fcntl.h>
#include <errno.h>
#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <inttypes.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <omp.h>

#include "core.h"
#include "../dataset_info.h"
#include "dataset.h"

static const char *split_name[] = {
    [SPLIT_TRAIN] = "train",
    [SPLIT_VALID] = "valid",
    [SPLIT_TEST]  = "test",
};

EdgeFormat parse_edge_format(const char* str)
{
    if (strncmp(str, "coo", 3) == 0) return EDGE_COO;
    if (strncmp(str, "csr", 3) == 0)return EDGE_CSR;
    if (strncmp(str, "csc", 3) == 0) return EDGE_CSC;
    ERROR("Not a valid edge format: %s", str);
}

static void path_join(char *buf, size_t size, const char *dir, const char *file)
{
    int ret;
    size_t dir_len = strlen(dir);
    if (dir[dir_len-1] != '/') ret = snprintf(buf, size, "%s/%s", dir, file);
    else ret = snprintf(buf, size, "%s%s", dir, file);

    if (ret < 0 || (size_t)ret >= size)
        ERROR("Path too long: %s/%s", dir, file);
}

static uint32_t load_split(const char *path, uint32_t **split)
{
    int fd = open(path, O_RDONLY);
    if (fd < 0) ERROR("Could not open %s: %s", path, strerror(errno));
    struct stat sb;
    fstat(fd, &sb);
    uint32_t* data = mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    size_t count = sb.st_size / sizeof(*data);
    *split = cache_aligned_alloc(sb.st_size);
#pragma omp parallel for
    for (size_t i = 0; i < count; i++)
    {
        (*split)[i] = data[i];
    }
    munmap(data, sb.st_size);
    close(fd);
    return count;
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

static double *load_nodes(const char *file)
{
    int fd = open(file, O_RDONLY);
    if (fd < 0) ERROR("Could not open %s: %s", file, strerror(errno));
    struct stat sb;
    fstat(fd, &sb);
    double *data = mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);
    size_t count = sb.st_size / sizeof(*data);
    double *nodes = cache_aligned_alloc(sb.st_size);
#pragma omp parallel for
    for(size_t i = 0; i < count; i++)
    {
        nodes[i] = data[i];
    }

    munmap(data, sb.st_size);
    close(fd);
    return nodes;
}

static uint32_t *load_labels(const char *file)
{
    int fd = open(file, O_RDONLY);
    if (fd < 0) ERROR("Could not open %s: %s", file, strerror(errno));
    struct stat sb;
    fstat(fd, &sb);
    uint32_t* data = mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    uint32_t count =  sb.st_size / sizeof(*data);
    uint32_t *labels = cache_aligned_alloc(sb.st_size);
#pragma omp parallel for
    for (size_t i = 0; i < count; i++)
    {
        labels[i] = data[i];
    }

    munmap(data, sb.st_size);
    close(fd);

    return labels;
}

typedef struct {
    uint32_t *u;
    uint32_t *v;
    uint32_t count; // only relevant in case of undirected graphs
} RawEdges;

static RawEdges load_edges(const char *file)
{
    int fd = open(file, O_RDONLY);
    if (fd < 0) ERROR("Could not open %s: %s", file, strerror(errno));
    struct stat sb;
    fstat(fd, &sb);
    uint32_t *data = mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    uint32_t count =  sb.st_size / sizeof(*data) / 2;

    uint32_t *u = cache_aligned_alloc(count * sizeof(*u));
    uint32_t *v = cache_aligned_alloc(count * sizeof(*v));

#pragma omp parallel for
    for (size_t i = 0; i < count; i++)
    {
        u[i] = data[2*i];
        v[i] = data[2*i+1];
    }

    munmap(data, sb.st_size);
    close(fd);
    return (RawEdges){ .u = u, .v = v, .count = count };
}

static int cmp_u64(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a;
    uint64_t y = *(const uint64_t *)b;
    return (x > y) - (x < y);
}

static void symmetrize_edges(RawEdges *raw)
{
    uint64_t *packed = cache_aligned_alloc(2 * raw->count * sizeof(*packed));
    uint32_t n = 0;

    for (uint32_t i = 0; i < raw->count; i++)
    {
        uint32_t u = raw->u[i];
        uint32_t v = raw->v[i];

        uint64_t fwd = ((uint64_t)u << 32) | v;
        packed[n++] = fwd;

        if (u != v)             // skip self-loops
        {
            uint64_t rev = ((uint64_t)v << 32) | u;
            packed[n++] = rev;
        }
    }

    qsort(packed, n, sizeof(uint64_t), cmp_u64);


    uint32_t unique = 0;
    for (uint32_t i = 0; i < n; i++)
    {
        if (unique == 0 || packed[i] != packed[unique - 1])
            packed[unique++] = packed[i];
    }

    raw->u = realloc(raw->u, unique * sizeof(*raw->u));
    raw->v = realloc(raw->v, unique * sizeof(*raw->v));
    raw->count = unique;
    for (uint32_t i = 0; i < unique; i++)
    {
        raw->u[i] = (uint32_t)(packed[i] >> 32);
        raw->v[i] = (uint32_t)(packed[i]);
    }

    free(packed);
}

static uint8_t *detect_self_loops(RawEdges raw_edges, uint32_t num_nodes)
{
    uint8_t *self_loop = cache_aligned_alloc(num_nodes * sizeof(*self_loop));
#pragma omp parallel for
    for (size_t i = 0; i < num_nodes; i++)
    {
        self_loop[i] = 0;
    }

    uint32_t self_loop_count = 0;
#pragma omp parallel for
    for (size_t i = 0; i < raw_edges.count; i++)
    {
        if (raw_edges.u[i] == raw_edges.v[i])
        {
            self_loop[raw_edges.u[i]] = 1;
            self_loop_count++;
        }
    }

    if (self_loop_count == 0)
    {
        printf("No self loops was found\n");
        free(self_loop);
        self_loop = NULL;
    }

    return self_loop;
}

typedef enum {
    IN_DEGREE,
    OUT_DEGREE,
} DegreeKind;

static Real *get_inv_degree(RawEdges raw_edges, uint32_t num_nodes, DegreeKind degree_kind)
{
    Real *inv_degree = cache_aligned_alloc(num_nodes * sizeof(*inv_degree));
    uint32_t *degree_count = cache_aligned_alloc(num_nodes * sizeof(*degree_count));

#pragma omp parallel for
    for (size_t i = 0; i < num_nodes; i++)
    {
        degree_count[i] = 0;
    }

    uint32_t *endpoints = NULL;
    if (degree_kind == IN_DEGREE)
    {
        endpoints = raw_edges.u;
    }
    else if (degree_kind == OUT_DEGREE)
    {
        endpoints = raw_edges.v;
    }
    else
    {
        ERROR("Missing DegreeKind");
    }

#pragma omp parallel for
    for (size_t i = 0; i < raw_edges.count; i++)
    {
        #pragma omp atomic
        degree_count[endpoints[i]] += 1;
    }

#pragma omp parallel for
    for (size_t i = 0; i < num_nodes; i++)
    {
        inv_degree[i] = 1.0 / degree_count[i];
    }

    free(degree_count);
    return inv_degree;
}


Dataset* dataset_load(char const* dataset, char const* data_dir, EdgeFormat format, bool to_symmetric)
{
    double t = omp_get_wtime();

    DatasetKind datatset_kind = str_to_dataset_kind(dataset);
    if (datatset_kind == DATASET_INVALID) ERROR("Given dataset is not valid: %s", dataset);

    // Equvalent of whats in nob.c
    DatasetInfo *ds_info = &ds_infos[datatset_kind];

    char ds_path[256];
    path_join(ds_path, sizeof(ds_path), data_dir, ds_info->dir_name);
    if (access(ds_path, F_OK) != 0)
    {
        ERROR("Dataset is missing, run ./nob -dataset %s -data_dir %s", dataset, data_dir);
    }
    char proc_path[256];
    path_join(proc_path, sizeof(proc_path), ds_path, "processed");

    Dataset *ds = malloc(sizeof(*ds));
    ds->path = malloc(strlen(ds_path)+1); strcpy(ds->path, ds_path);
    ds->num_nodes = ds_info->num_nodes;
    ds->num_features = ds_info->num_features;
    ds->num_classes = ds_info->num_classes;
    ds->num_edges = ds_info->num_edges;

    char bin_path[256];

    // Edges
    double t_e = omp_get_wtime();
    path_join(bin_path, sizeof(bin_path), proc_path, "edge.bin");
    RawEdges raw_edges = load_edges(bin_path);

    ds->edges.self_loop = detect_self_loops(raw_edges, ds->num_nodes);
    ds->edges.inv_in_degree = get_inv_degree(raw_edges, ds->num_nodes, IN_DEGREE);
    ds->edges.inv_out_degree = get_inv_degree(raw_edges, ds->num_nodes, OUT_DEGREE);

    if (to_symmetric || ds_info->undirected)
    {
        symmetrize_edges(&raw_edges);
        ds->num_edges = raw_edges.count;
    }

    ds->edges.format = format;
    switch(format)
    {
    case EDGE_COO:
        ds->edges.src = cache_aligned_alloc(ds->num_edges*sizeof(*ds->edges.src));
        ds->edges.dst = cache_aligned_alloc(ds->num_edges*sizeof(*ds->edges.dst));
        if (!ds->edges.src || !ds->edges.dst) ERROR("Could not allocate COO edges");
        // First touch
#pragma omp parallel for
        for (size_t i = 0; i < raw_edges.count; i++)
        {
            ds->edges.src[i] = raw_edges.u[i];
            ds->edges.dst[i] = raw_edges.v[i];
        }
        break;
    case EDGE_CSR:
        TODO("Implement CSR convertion");
        break;
    case EDGE_CSC:
        TODO("Implement CSC convertion");
        break;
    default:
        ERROR("Unknown edge format type %d", format);
    }
    free(raw_edges.u);
    free(raw_edges.v);
    t_e = omp_get_wtime() - t_e;

    double t_l = omp_get_wtime();
    path_join(bin_path, sizeof(bin_path), proc_path, "node-label.bin");
    ds->labels = load_labels(bin_path);
    t_l = omp_get_wtime() - t_l;

    double t_f = omp_get_wtime();
    path_join(bin_path, sizeof(bin_path), proc_path, "node-feat.bin");
    ds->nodes = load_nodes(bin_path);
    t_f = omp_get_wtime() - t_f;

    printf("Loaded OGB-ARXIV in %.2fs\n", omp_get_wtime() - t);
    printf("    loading edges: %.2fs\n", t_e);
    printf("    loading features: %.2fs\n", t_f);
    printf("    loading labels: %.2fs\n", t_l);

    return ds;
}

#define INVALID_IDX UINT32_MAX

Dataset *dataset_split(Dataset *base, Split split)
{
    double t = omp_get_wtime();
    uint32_t *split_idx, split_size;

    char proc_path[256];
    path_join(proc_path, sizeof(proc_path), base->path, "processed");

    const char *split_names[] = {
        [SPLIT_TRAIN] = "train.bin",
        [SPLIT_VALID] = "valid.bin",
        [SPLIT_TEST]  = "test.bin",
    };

    if (split < 0 || split > SPLIT_TEST)
    {
        ERROR("Unknown split: %d", split);
    }

    char split_file[256];
    path_join(split_file, sizeof(split_file), proc_path, split_names[split]);
    split_size = load_split(split_file, &split_idx);

#if defined(REDUCED_GRAPH)
    static const uint32_t reduced_sizes[] = {
        [SPLIT_TRAIN] = 700,
        [SPLIT_VALID] = 400,
        [SPLIT_TEST]  = 600,
    };
    split_size = reduced_sizes[split];
#endif

    Dataset *ds = malloc(sizeof(*ds));
    if (!ds) ERROR("Could not allocate split Dataset");
    ds->path = malloc(strlen(base->path)+1); strcpy(ds->path, base->path);
    ds->num_features = base->num_features;
    ds->num_classes  = base->num_classes;

    // Gather features
    ds->nodes = cache_aligned_alloc(split_size * ds->num_features * sizeof(*ds->nodes));
#pragma omp parallel for
    for (size_t i = 0; i < split_size; i++) {
        memcpy(&ds->nodes[i * ds->num_features],
               &base->nodes[split_idx[i] * ds->num_features],
               ds->num_features * sizeof(*ds->nodes));
    }

    // Gather labels
    ds->labels = cache_aligned_alloc(split_size * sizeof(*ds->labels));
    for (size_t i = 0; i < split_size; i++) {
        ds->labels[i] = base->labels[split_idx[i]];
    }

    uint32_t *node_map = cache_aligned_alloc(base->num_nodes * sizeof(*node_map));
    build_node_mapping(node_map, base->num_nodes, split_idx, split_size);

    // Count number of edges in the split
    // We do this in two passes since realloc might break cache allignment
    uint32_t edge_count = 0;
    for (uint32_t i = 0; i < base->num_edges; i++)
    {
        uint32_t src = node_map[base->edges.src[i]];
        uint32_t dst = node_map[base->edges.dst[i]];
        if (src != INVALID_IDX && dst != INVALID_IDX)
        {
            edge_count++;
        }
    }
    ds->edges.src = cache_aligned_alloc(edge_count * sizeof(*ds->edges.src));
    ds->edges.dst = cache_aligned_alloc(edge_count * sizeof(*ds->edges.dst));
    uint32_t edge_idx = 0;
    for (uint32_t i = 0; i < base->num_edges; i++)
    {
        uint32_t src = node_map[base->edges.src[i]];
        uint32_t dst = node_map[base->edges.dst[i]];
        if (src != INVALID_IDX && dst != INVALID_IDX)
        {
            ds->edges.src[edge_idx] = src;
            ds->edges.dst[edge_idx] = dst;
            edge_idx++;
        }
    }

    // Transfer base's self loop
    RawEdges raw_edges = { .u = ds->edges.src, .v = ds->edges.dst, .count = edge_count };
    ds->edges.self_loop = detect_self_loops(raw_edges, split_size);
    ds->edges.inv_in_degree = get_inv_degree(raw_edges, split_size, IN_DEGREE);
    ds->edges.inv_out_degree = get_inv_degree(raw_edges, split_size, OUT_DEGREE);

    ds->num_nodes = split_size;
    ds->num_edges  = edge_count;

    free(node_map);
    free(split_idx);

    printf("Split OGB-ARXIV [%s] in %.2fs\n", split_name[split], omp_get_wtime() - t);
    return ds;
}

void dataset_free(Dataset **ds)
{
    if (!(*ds)) return;

    free((*ds)->edges.src); (*ds)->edges.src = NULL;
    free((*ds)->edges.dst); (*ds)->edges.dst = NULL;
    free((*ds)->path);      (*ds)->path      = NULL;
    free((*ds)->nodes);     (*ds)->nodes     = NULL;
    free((*ds)->labels);    (*ds)->labels    = NULL;
    free((*ds));
    *ds = NULL;
}
