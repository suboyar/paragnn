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
#include "dataset.h"
#include "dataset_info.h"

#define INVALID_IDX -1          // assumes signed node indecies

static const char *split_name[] = {
    [SPLIT_TRAIN] = "train",
    [SPLIT_VALID] = "valid",
    [SPLIT_TEST]  = "test",
};

EdgeFormat parse_edge_format(const char* str)
{
    if (strcmp(str, "coo") == 0)        return EDGE_COO;
    if (strcmp(str, "compressed") == 0) return EDGE_CSX;
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

static int64_t load_split(const char *path, int64_t **split)
{
    int fd = open(path, O_RDONLY);
    if (fd < 0) ERROR("Could not open %s: %s", path, strerror(errno));
    struct stat sb;
    fstat(fd, &sb);
    int64_t* data = mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
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

static void build_node_mapping(int64_t* map,  int64_t num_nodes, int64_t* split_idx, int64_t split_size)
{
    memset(map, 0xFF, num_nodes * sizeof(*map));

    for (int64_t i = 0; i < split_size; i++)
    {
        map[split_idx[i]] = i;
    }
}

static Real *load_nodes(const char *file)
{
    int fd = open(file, O_RDONLY);
    if (fd < 0) ERROR("Could not open %s: %s", file, strerror(errno));
    struct stat sb;
    fstat(fd, &sb);
    double *data = mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);
    size_t count = sb.st_size / sizeof(*data);
    Real *nodes = cache_aligned_alloc(sb.st_size);
#pragma omp parallel for
    for(size_t i = 0; i < count; i++)
    {
        nodes[i] = (Real)data[i];
    }

    munmap(data, sb.st_size);
    close(fd);
    return nodes;
}

static int64_t *load_labels(const char *file)
{
    int fd = open(file, O_RDONLY);
    if (fd < 0) ERROR("Could not open %s: %s", file, strerror(errno));
    struct stat sb;
    fstat(fd, &sb);
    int64_t* data = mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    size_t count =  sb.st_size / sizeof(*data);
    int64_t *labels = cache_aligned_alloc(sb.st_size);
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
    int64_t *u;
    int64_t *v;
    int64_t count; // only relevant in case of undirected graphs
} RawEdges;

static RawEdges load_edges(const char *file)
{
    int fd = open(file, O_RDONLY);
    if (fd < 0) ERROR("Could not open %s: %s", file, strerror(errno));
    struct stat sb;
    fstat(fd, &sb);
    int64_t *data = mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    int64_t count =  sb.st_size / sizeof(*data) / 2;

    int64_t *u = cache_aligned_alloc(count * sizeof(*u));
    int64_t *v = cache_aligned_alloc(count * sizeof(*v));

#pragma omp parallel for
    for (int64_t i = 0; i < count; i++)
    {
        u[i] = data[2*i];
        v[i] = data[2*i+1];
    }

    munmap(data, sb.st_size);
    close(fd);
    return (RawEdges){ .u = u, .v = v, .count = count };
}

static int cmp_s128(const void *a, const void *b) {
    signed __int128 x = *(const signed __int128 *)a;
    signed __int128 y = *(const signed __int128 *)b;
    return (x > y) - (x < y);
}

static void symmetrize_edges(RawEdges *raw)
{
    signed __int128 *packed = cache_aligned_alloc(2 * raw->count * sizeof(*packed));
    int64_t n = 0;

    for (int64_t i = 0; i < raw->count; i++)
    {
        int64_t u = raw->u[i];
        int64_t v = raw->v[i];

        signed __int128 fwd = ((signed __int128)u << 64) | (uint64_t)v;
        packed[n++] = fwd;

        if (u != v)             // skip self-loops
        {
            signed __int128 rev = ((signed __int128)v << 64) | (uint64_t)u;
            packed[n++] = rev;
        }
    }

    qsort(packed, n, sizeof(*packed), cmp_s128);


    int64_t unique = 0;
    for (int64_t i = 0; i < n; i++)
    {
        if (unique == 0 || packed[i] != packed[unique - 1])
            packed[unique++] = packed[i];
    }

    raw->u = realloc(raw->u, unique * sizeof(*raw->u));
    raw->v = realloc(raw->v, unique * sizeof(*raw->v));
    raw->count = unique;
    for (int64_t i = 0; i < unique; i++)
    {
        raw->u[i] = (int64_t)(packed[i] >> 64);
        raw->v[i] = (int64_t)(packed[i]);
    }

    free(packed);
}

static uint8_t *detect_self_loops_coo(const int64_t *restrict nodes, const int64_t *restrict peers, int64_t num_nodes, int64_t num_edges)
{
    uint8_t *self_loop = cache_aligned_alloc(num_nodes * sizeof(*self_loop));

#pragma omp parallel for
    for (int64_t i = 0; i < num_nodes; i++)
    {
        self_loop[i] = 0;
    }

    int64_t self_loop_count = 0;
#pragma omp parallel for
    for (int64_t i = 0; i < num_edges; i++)
    {
        if (nodes[i] == peers[i])
        {
            self_loop[nodes[i]] = 1;
            self_loop_count++;
        }
    }

    if (self_loop_count == 0)
    {
        free(self_loop);
        self_loop = NULL;
    }

    return self_loop;
}

static uint8_t *detect_self_loops_crx(int64_t *ptr, int64_t *idx, int64_t num_nodes)
{
    uint8_t *self_loop = cache_aligned_alloc(num_nodes * sizeof(*self_loop));
    int64_t self_loop_count = 0;
#pragma omp parallel for
    for (int64_t i = 0; i < num_nodes; i++)
    {
        self_loop[i] = 0;
        for (int64_t j = ptr[i]; j < ptr[i+1]; j++)
        {
            if (idx[j] == i)
            {
                self_loop[i] = 1;
                self_loop_count++;
                break;
            }
        }
    }

    if (self_loop_count == 0)
    {
        free(self_loop);
        self_loop = NULL;
    }

    return self_loop;
}

typedef enum {
    IN_DEGREE,
    OUT_DEGREE,
} DegreeKind;

static Real *get_inv_degree_coo(const int64_t *restrict nodes, const int64_t *restrict peers,
                                int64_t num_nodes, int64_t num_edges, DegreeKind degree_kind)
{
    Real *inv_degree = cache_aligned_alloc(num_nodes * sizeof(*inv_degree));
    int64_t *degree_count = cache_aligned_alloc(num_nodes * sizeof(*degree_count));

#pragma omp parallel for
    for (int64_t i = 0; i < num_nodes; i++)
    {
        degree_count[i] = 0;
        inv_degree[i] = 0;
    }

    const int64_t *endpoints = NULL;
    if (degree_kind == IN_DEGREE)
    {
        endpoints = peers;
    }
    else // degree_kind == OUT_DEGREE
    {
        endpoints = nodes;
    }

#pragma omp parallel for
    for (int64_t i = 0; i < num_edges; i++)
    {
        #pragma omp atomic
        degree_count[endpoints[i]] += 1;
    }

#pragma omp parallel for
    for (int64_t i = 0; i < num_nodes; i++)
    {
        inv_degree[i] = REAL(1.0) / degree_count[i];
    }

    free(degree_count);
    return inv_degree;
}

static Real *get_inv_degree_crx(int64_t *ptr, int64_t num_nodes)
{
    Real *inv_degree = cache_aligned_alloc(num_nodes * sizeof(*inv_degree));

#pragma omp parallel for
    for (int64_t i = 0; i < num_nodes; i++)
    {
        inv_degree[i] = REAL(1.0) / (ptr[i+1] - ptr[i]);
    }

    return inv_degree;
}

enum CSX_TYPE {CSR, CSC};

static void raw2crx(int64_t **ptr, int64_t **idx, RawEdges raw_edges, int64_t num_nodes, int64_t num_edges, enum CSX_TYPE csx_type)
{
    *ptr = cache_aligned_alloc((num_nodes+1) * sizeof(**ptr));
    // NUMA first touch
#pragma omp parallel for
    for (int64_t i = 0; i < (num_nodes+1); i++)
    {
        (*ptr)[i] = 0;
    }
    *idx = malloc(num_edges * sizeof(**idx));
#pragma omp parallel for
    for (int64_t i = 0; i < num_edges; i++)
    {
        (*idx)[i] = 0;
    }

    int64_t *pos = malloc(num_nodes * sizeof(*pos));
    int64_t *major, *minor;

    if (csx_type == CSR)
    {
        major = raw_edges.u;
        minor = raw_edges.v;
    }
    else  // csx_type == CSC
    {
        major = raw_edges.v;
        minor = raw_edges.u;
    }

    for (int64_t i = 0; i < num_edges; i++)
    {
        (*ptr)[major[i]+1]++;
    }

    for(int64_t i = 1; i < (num_nodes+1); i++)
    {
        (*ptr)[i] += (*ptr)[i-1];
    }

    for (int64_t i = 0; i < num_nodes; i++)
    {
        pos[i] = (*ptr)[i];
    }

    for (int64_t i = 0; i < num_edges; i++)
    {
        (*idx)[pos[major[i]]++] = minor[i];
    }

    free(pos);
}

Dataset* dataset_load(DatasetKind dataset, char const* datadir, EdgeFormat format)
{
    double t = omp_get_wtime();

    if (dataset == DATASET_INVALID) ERROR("Given dataset kind is not valid: %d", dataset);

    // Equvalent of whats in nob.c
    DatasetInfo ds_info = ds_infos[dataset];

    char ds_path[256];
    path_join(ds_path, sizeof(ds_path), datadir, ds_info.dir_name);
    if (access(ds_path, F_OK) != 0)
    {
        ERROR("Dataset is missing, run ./nob -dataset %s -datadir %s", ds_info.name, datadir);
    }
    char proc_path[256];
    path_join(proc_path, sizeof(proc_path), ds_path, "processed");

    Dataset *ds = malloc(sizeof(*ds));
    ds->path = malloc(strlen(ds_path)+1); strcpy(ds->path, ds_path);
    ds->num_nodes     = ds_info.num_nodes;
    ds->num_features  = ds_info.num_features;
    ds->num_classes   = ds_info.num_classes;
    ds->num_edges     = ds_info.num_edges;
    ds->edges.format = format;
    ds->edges.src     = NULL;
    ds->edges.dst     = NULL;
    ds->edges.ptr_csc = NULL;
    ds->edges.idx_csc = NULL;
    ds->edges.ptr_csr = NULL;
    ds->edges.idx_csr = NULL;

    char bin_path[256];

    // Edges
    double t_e = omp_get_wtime();
    path_join(bin_path, sizeof(bin_path), proc_path, "edge.bin");
    RawEdges raw_edges = load_edges(bin_path);
    if (ds_info.directed)
    {
        symmetrize_edges(&raw_edges);
    }
    ds->num_edges = raw_edges.count;

    switch(format)
    {
    case EDGE_COO:
    {
        ds->edges.src = cache_aligned_alloc(ds->num_edges*sizeof(*ds->edges.src));
        ds->edges.dst = cache_aligned_alloc(ds->num_edges*sizeof(*ds->edges.dst));
        if (!ds->edges.src || !ds->edges.dst) ERROR("Could not allocate COO edges");
        // First touch
#pragma omp parallel for
        for (int64_t i = 0; i < ds->num_edges; i++)
        {
            ds->edges.src[i] = raw_edges.u[i];
            ds->edges.dst[i] = raw_edges.v[i];
        }
        ds->edges.self_loop = detect_self_loops_coo(ds->edges.src, ds->edges.dst, ds->num_nodes, ds->num_edges);
        ds->edges.inv_in_degree  = get_inv_degree_coo(ds->edges.src, ds->edges.dst, ds->num_nodes, ds->num_edges, IN_DEGREE);
        ds->edges.inv_out_degree = get_inv_degree_coo(ds->edges.src, ds->edges.dst, ds->num_nodes, ds->num_edges, OUT_DEGREE);
        break;
    }
    case EDGE_CSX:
        raw2crx(&ds->edges.ptr_csc, &ds->edges.idx_csc, raw_edges, ds->num_nodes, ds->num_edges, CSC);
        raw2crx(&ds->edges.ptr_csr, &ds->edges.idx_csr, raw_edges, ds->num_nodes, ds->num_edges, CSR);
        ds->edges.self_loop = detect_self_loops_crx(ds->edges.ptr_csc, ds->edges.idx_csc, ds->num_nodes);
        ds->edges.inv_in_degree = NULL;
        ds->edges.inv_out_degree = NULL;
        break;
    default:
        ERROR("Invalid edge format type %d", format);
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

    // Compute statistics
    ds->edges.avg_degree = (float)ds->num_edges / ds->num_nodes;
    int64_t self_loop_count = 0;
    for (int64_t i = 0; ds->edges.self_loop && i < ds->num_nodes; i++)
    {
        if (ds->edges.self_loop[i]) self_loop_count++;
    }
    ds->edges.avg_self_loop = (float)self_loop_count / ds->num_nodes;

    printf("Loaded OGB-ARXIV in %.2fs (node count: %ld, edge count: %ld, avg degree: %.2f, avg self loops: %.2f)\n",
           omp_get_wtime() - t, ds->num_nodes, ds->num_edges, ds->edges.avg_degree, ds->edges.avg_self_loop);
    printf("    loading edges: %.2fs\n", t_e);
    printf("    loading features: %.2fs\n", t_f);
    printf("    loading labels: %.2fs\n", t_l);

    return ds;
}

void split_csx(Edges *edges, Edges base_edges, const int64_t *restrict node_map, int64_t num_nodes, int64_t base_num_nodes, enum CSX_TYPE csx_type)
{
    const int64_t *restrict base_ptr, *restrict base_idx;
    if (csx_type == CSC)
    {
        base_ptr = base_edges.ptr_csc;
        base_idx = base_edges.idx_csc;
    }
    else // csx_type == CSR
    {
        base_ptr = base_edges.ptr_csr;
        base_idx = base_edges.idx_csr;
    }

    int64_t *restrict ptr, *restrict idx;
    ptr = cache_aligned_alloc((num_nodes + 1) * sizeof(*ptr));
    // For NUMA first touch
#pragma omp parallel for
    for (int64_t i = 0; i < num_nodes + 1; i++)
    {
        ptr[i] = 0;
    }

    for (int64_t i = 0; i < base_num_nodes; i++)
    {
        int64_t m = node_map[i];
        if (m == INVALID_IDX) continue;
        for (int64_t j =  base_ptr[i]; j < base_ptr[i+1]; j++)
        {
            if (node_map[base_idx[j]] != INVALID_IDX)
            {
                ptr[m + 1]++;
            }
        }
    }

    for (int64_t i = 1; i <= num_nodes; i++)
    {
        ptr[i] += ptr[i - 1];
    }

    int64_t num_edges = ptr[num_nodes];
    int64_t *pos = malloc(num_nodes * sizeof(*pos));
    memcpy(pos, ptr, num_nodes * sizeof(*pos));
    idx = cache_aligned_alloc(num_edges * sizeof(*idx));
    // For NUMA first touch
#pragma omp parallel for
    for (int64_t i = 0; i < num_edges; i++)
    {
        idx[i] = 0;
    }

    for (int64_t i = 0; i < base_num_nodes; i++)
    {
        int64_t m = node_map[i];
        if (m == INVALID_IDX) continue;
        for (int64_t j = base_ptr[i]; j < base_ptr[i+1]; j++)
        {
            int64_t n = node_map[base_idx[j]];
            if (n == INVALID_IDX) continue;
            idx[pos[m]++] = n;
        }
    }

    if (csx_type == CSC)
    {
        edges->ptr_csc = ptr;
        edges->idx_csc = idx;
    }
    else
    {
        edges->ptr_csr = ptr;
        edges->idx_csr = idx;
    }

    free(pos);
}

Dataset *dataset_split(Dataset *base, Split split)
{
    double t = omp_get_wtime();
    int64_t *split_idx, split_size;

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
    static const int64_t reduced_sizes[] = {
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
    ds->num_nodes = split_size;
    ds->edges.format = base->edges.format;
    ds->edges.src     = NULL;
    ds->edges.dst     = NULL;
    ds->edges.ptr_csc = NULL;
    ds->edges.idx_csc = NULL;
    ds->edges.ptr_csr = NULL;
    ds->edges.idx_csr = NULL;

    // Gather features
    ds->nodes = cache_aligned_alloc(ds->num_nodes * ds->num_features * sizeof(*ds->nodes));
#pragma omp parallel for
    for (int64_t i = 0; i < ds->num_nodes; i++) {
        memcpy(&ds->nodes[i * ds->num_features],
               &base->nodes[split_idx[i] * ds->num_features],
               ds->num_features * sizeof(*ds->nodes));
    }

    // Gather labels
    ds->labels = cache_aligned_alloc(ds->num_nodes * sizeof(*ds->labels));
    for (int64_t i = 0; i < ds->num_nodes; i++) {
        ds->labels[i] = base->labels[split_idx[i]];
    }

    int64_t *node_map = cache_aligned_alloc(base->num_nodes * sizeof(*node_map));
    build_node_mapping(node_map, base->num_nodes, split_idx, ds->num_nodes);

    ds->num_edges = 0;      // Counting happens in the switch statement
    switch(ds->edges.format)
    {
    case EDGE_COO:
    {
        // Count number of edges in the split
        // We do this in two passes since realloc might break cache allignment
        for (int64_t i = 0; i < base->num_edges; i++)
        {
            int64_t src = node_map[base->edges.src[i]];
            int64_t dst = node_map[base->edges.dst[i]];
            if (src != INVALID_IDX && dst != INVALID_IDX)
            {
                ds->num_edges++;
            }
        }
        int64_t ei = 0;
        ds->edges.src = cache_aligned_alloc(ds->num_edges * sizeof(*ds->edges.src));
        ds->edges.dst = cache_aligned_alloc(ds->num_edges * sizeof(*ds->edges.dst));
        for (int64_t i = 0; i < base->num_edges; i++)
        {
            int64_t src = node_map[base->edges.src[i]];
            int64_t dst = node_map[base->edges.dst[i]];
            if (src != INVALID_IDX && dst != INVALID_IDX)
            {
                ds->edges.src[ei] = src;
                ds->edges.dst[ei] = dst;
                ei++;
            }
        }

        ds->edges.self_loop = detect_self_loops_coo(ds->edges.src, ds->edges.dst, ds->num_nodes, ds->num_edges);
        ds->edges.inv_in_degree = get_inv_degree_coo(ds->edges.src, ds->edges.dst, ds->num_nodes, ds->num_edges, IN_DEGREE);
        ds->edges.inv_out_degree = get_inv_degree_coo(ds->edges.src, ds->edges.dst, ds->num_nodes, ds->num_edges, OUT_DEGREE);
        break;
    }
    case EDGE_CSX:
    {
        split_csx(&ds->edges, base->edges, node_map, ds->num_nodes, base->num_nodes, CSC);
        split_csx(&ds->edges, base->edges, node_map, ds->num_nodes, base->num_nodes, CSR);
        ds->num_edges = ds->edges.ptr_csc[ds->num_nodes];
        ds->edges.self_loop = detect_self_loops_crx(ds->edges.ptr_csc, ds->edges.idx_csc, ds->num_nodes);
        ds->edges.inv_in_degree = NULL;
        ds->edges.inv_out_degree = NULL;
        break;
    }
    default:
        ERROR("Invalid edge format type %d", base->edges.format);
    }

    // Compute statistics
    ds->edges.avg_degree = (float)ds->num_edges / ds->num_nodes;
    int64_t self_loop_count = 0;
    for (int64_t i = 0; ds->edges.self_loop && i < ds->num_nodes; i++)
    {
        if (ds->edges.self_loop[i]) self_loop_count++;
    }
    ds->edges.avg_self_loop = (float)self_loop_count / ds->num_nodes;

    printf("Split OGB-ARXIV [%s] in %.2fs (node count: %ld, edge count %ld, avg degree: %.2f, avg self loops: %.2f)\n",
           split_name[split], omp_get_wtime() - t, ds->num_nodes, ds->num_edges, ds->edges.avg_degree, ds->edges.avg_self_loop);

    free(node_map);
    free(split_idx);
    return ds;
}

void dataset_free(Dataset **ds)
{
    if (!(*ds)) return;

    // Free edges
    free((*ds)->edges.src);            (*ds)->edges.src            = NULL;
    free((*ds)->edges.dst);            (*ds)->edges.dst            = NULL;
    free((*ds)->edges.ptr_csc);        (*ds)->edges.ptr_csc        = NULL;
    free((*ds)->edges.idx_csc);        (*ds)->edges.idx_csc        = NULL;
    free((*ds)->edges.ptr_csr);        (*ds)->edges.ptr_csr        = NULL;
    free((*ds)->edges.idx_csr);        (*ds)->edges.idx_csr        = NULL;
    free((*ds)->edges.self_loop);      (*ds)->edges.self_loop      = NULL;
    free((*ds)->edges.inv_in_degree);  (*ds)->edges.inv_in_degree  = NULL;
    free((*ds)->edges.inv_out_degree); (*ds)->edges.inv_out_degree = NULL;

    free((*ds)->path);   (*ds)->path   = NULL;
    free((*ds)->nodes);  (*ds)->nodes  = NULL;
    free((*ds)->labels); (*ds)->labels = NULL;
    free((*ds));
    *ds = NULL;
}
