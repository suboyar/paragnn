#include "ds.h"

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <fcntl.h>
#include <inttypes.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <omp.h>

#include "core.h"
#include "dsinfo.h"
#include "timer.h"

#define INVALID_IDX -1          // assumes signed node indecies

static const char *split_name[] = {
    [SPLIT_TRAIN] = "train",
    [SPLIT_VALID] = "valid",
    [SPLIT_TEST]  = "test",
};

SparseFormat parse_edge_format(const char* str)
{
    if (strcmp(str, "coo") == 0)        return SPARSE_COO;
    if (strcmp(str, "compressed") == 0) return SPARSE_CS;
    ERROR("Not a valid edge format: %s", str);
}

static char *temp_buf = NULL;
static size_t temp_cap = 0;
static size_t temp_used = 0;

static void temp_reset(void) { temp_used = 0; }
static void temp_free(void) { free(temp_buf); temp_buf = NULL; temp_cap = 0; temp_used = 0; }
static char *temp_alloc(size_t n)
{
    if (temp_used + n > temp_cap)
    {
        if (temp_cap == 0) temp_cap = 4096;
        while (temp_used + n > temp_cap) temp_cap *= 2;
        temp_buf = ALLOC_OR_DIE(realloc(temp_buf, temp_cap));
    }
    char *p = temp_buf + temp_used;
    temp_used += n;
    return p;
}

static char *path_join(const char *dir, const char *file)
{
    size_t dir_len = strlen(dir);
    size_t file_len = strlen(file);
    bool need_sep = (dir_len > 0 && dir[dir_len - 1] != '/');
    size_t total = dir_len + need_sep + file_len + 1;
    char *out = temp_alloc(total);

    memcpy(out, dir, dir_len);
    if (need_sep) out[dir_len] = '/';
    memcpy(out + dir_len + need_sep, file, file_len);
    out[total - 1] = '\0';
    return out;
}

static int64_t count_bin_elements(const char *path)
{
    struct stat sb;
    if (stat(path, &sb) != 0) ERROR("stat failed for %s: %s", path, strerror(errno));

    return sb.st_size / sizeof(int64_t);
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

#define INVALID_IDX -1          // assumes signed node indecies

static void build_node_mapping(int64_t* map,  int64_t num_nodes, int64_t* split_idx, int64_t split_size)
{
    memset(map, 0xFF, num_nodes * sizeof(*map));

    for (int64_t i = 0; i < split_size; i++)
    {
        map[split_idx[i]] = i;
    }
}

static size_t get_file_size(const char *file)
{
    int fd = open(file, O_RDONLY);
    if (fd < 0) ERROR("Could not open %s: %s", file, strerror(errno));
    struct stat sb;
    fstat(fd, &sb);
    close(fd);
    return sb.st_size;
}

static void load_feats(const char *file, Real *dest)
{
    MmapInfo info = map_file(file, PROT_READ, MAP_PRIVATE | MAP_POPULATE);
    double *data = (double*)info.data;
    size_t count = info.bytes / sizeof(*data);
#pragma omp parallel for
    for(size_t i = 0; i < count; i++)
    {
        dest[i] = (Real)data[i];
    }
    unmap_file(&info);
}

static int64_t *load_labels(const char *file, int64_t *dest)
{
    MmapInfo info = map_file(file, PROT_READ, MAP_PRIVATE | MAP_POPULATE);
    int64_t *data = (int64_t*)info.data;
    size_t count = info.bytes / sizeof(*data);
#pragma omp parallel for
    for (size_t i = 0; i < count; i++)
    {
        dest[i] = data[i];
    }
    unmap_file(&info);
}

const char* dataset_split_name(Dataset *ds)
{
    static const char *strings[SPLIT_COUNT] = {
        [SPLIT_NONE] = "full",
        [SPLIT_TRAIN] = "train",
        [SPLIT_VALID] = "valid",
        [SPLIT_TEST] = "test",
    };

    if (ds->split <= SPLIT_INVALID || ds->split >= SPLIT_COUNT) ERROR("Invlid split value: %d\n", ds->split);
    return strings[ds->split];
}


Dataset* dataset_alloc(DatasetKind dskind, char const *root, SparseFormat format, Split split)
{
    Dataset *ds = ALLOC_OR_DIE(calloc(1, sizeof(*ds)));

    if (dskind <= DATASET_INVALID || dskind >= DATASET_COUNT) ERROR("Given dataset kind is not valid: %d", dskind);
    ds->info = &ds_infos[dskind];
    char *ds_path = path_join(root, ds->info->dir_name);
    if (access(ds_path, F_OK) != 0) ERROR("Dataset is missing, run:\n  dsprep --dataset %s --root %s", ds->info->name, root);

    char *label_bin_path, *feat_bin_path, *edge_bin_path;
    switch (split)
    {
        case SPLIT_NONE:
            label_bin_path = path_join(ds_path, "processed/node-label.bin");
            feat_bin_path  = path_join(ds_path, "processed/node-feat.bin");
            edge_bin_path  = path_join(ds_path, "processed/edge.bin");
            break;
        case SPLIT_TRAIN:
            label_bin_path = path_join(ds_path, "processed/train-node-label.bin");
            feat_bin_path  = path_join(ds_path, "processed/train-node-feat.bin");
            edge_bin_path  = path_join(ds_path, "processed/train-edge.bin");
            break;
        case SPLIT_VALID:
            label_bin_path = path_join(ds_path, "processed/valid-node-label.bin");
            feat_bin_path  = path_join(ds_path, "processed/valid-node-feat.bin");
            edge_bin_path  = path_join(ds_path, "processed/valid-edge.bin");
            break;
        case SPLIT_TEST:
            label_bin_path = path_join(ds_path, "processed/test-node-label.bin");
            feat_bin_path  = path_join(ds_path, "processed/test-node-feat.bin");
            edge_bin_path  = path_join(ds_path, "processed/test-edge.bin");
            break;
        default:
            UNREACHABLE("Unknown Split enum value %d", split);
    }
    int64_t node_count = count_bin_elements(label_bin_path);
    int64_t edge_count = count_bin_elements(edge_bin_path) / 2;
    int64_t feature_count = ds->info->feature_count;
    int64_t class_count = ds->info->class_count;

    ds->label_path    = ALLOC_OR_DIE(strdup(label_bin_path));
    ds->feat_path     = ALLOC_OR_DIE(strdup(feat_bin_path));
    ds->node_count    = node_count;
    ds->feature_count = feature_count;
    ds->class_count   = class_count;
    ds->edge_count    = edge_count;
    ds->split         = split;
    ds->nodes         = ALLOC_OR_DIE(cache_aligned_alloc(node_count * feature_count * sizeof(*ds->nodes)));
    ds->labels        = ALLOC_OR_DIE(cache_aligned_alloc(node_count * sizeof(*ds->labels)));
    ds->graph         = sparsegraph_alloc(node_count, edge_count, ds->info->add_inverse_edge, edge_bin_path, format);

    temp_free();
    return ds;
}

void dataset_load_ex(Dataset *ds, bool verbose)
{
    double start_time, label_time, feat_time, edge_time;

    if (!ds) ERROR("Dataset has not been allocated, dataset_alloc needs to be run first");

    TIMER_NORECORD(label_time, load_labels(ds->label_path, ds->labels));
    TIMER_NORECORD(feat_time, load_feats(ds->feat_path, ds->nodes));
    TIMER_NORECORD(edge_time, sparsegraph_load(ds->graph));

    if (verbose)
    {
        printf("Loaded %s [%s] (nodes: %ld, edges: %ld) | Times: label %.2fs, feat %.2fs, edge %.2fs\n",
               ds->info->name, dataset_split_name(ds), ds->node_count, ds->edge_count,
               label_time, feat_time, edge_time);
    }
}

void dataset_load(Dataset *ds)
{
    dataset_load_ex(ds, true);
}

Dataset* dataset_alloc_and_load(DatasetKind dskind, char const *root, SparseFormat format, Split split)
{
    Dataset* ds = dataset_alloc(dskind, root, format, split);
    dataset_load(ds);
    return ds;
}

void dataset_reset_alloc(Dataset *ds)
{
    free(ds->nodes);
    ds->nodes = ALLOC_OR_DIE(cache_aligned_alloc(ds->node_count * ds->feature_count * sizeof(*ds->nodes)));
    free(ds->labels);
    ds->labels = ALLOC_OR_DIE(cache_aligned_alloc(ds->node_count * sizeof(*ds->labels)));
    sparsegraph_reset_alloc(ds->graph);
}

void dataset_free(Dataset **ds)
{
    if (!*ds) return;
    free((*ds)->label_path);
    free((*ds)->feat_path);
    free((*ds)->nodes);
    free((*ds)->labels);
    sparsegraph_free(&(*ds)->graph);
    free(*ds);
    *ds = NULL;
}
