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

SparseFormat parse_edge_format(const char* str)
{
    if (strcmp(str, "coo") == 0)        return SPARSE_FORMAT_COO;
    if (strcmp(str, "compressed") == 0) return SPARSE_FORMAT_CS;
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
    *split = alloc_local(sb.st_size);
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < count; i++)
        (*split)[i] = data[i];
    munmap(data, sb.st_size);
    close(fd);
    return count;
}

#define INVALID_IDX -1          // assumes signed node indecies

static void create_split_mask(int64_t node_count, const int64_t *full_labels, const char *split_path, int64_t *mask)
{
    for (int64_t i = 0; i < node_count; i++)
        mask[i] = INVALID_IDX;

    int64_t *split_idx = NULL;
    int64_t count = load_split(split_path, &split_idx);

    for (int64_t i = 0; i < count; i++)
        mask[split_idx[i]] = full_labels[split_idx[i]];

    free(split_idx);
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
#pragma omp parallel for schedule(static)
    for(size_t i = 0; i < count; i++)
    {
        dest[i] = (Real)data[i];
    }
    unmap_file(&info);
}

static void load_labels(const char *file, int64_t *dest)
{
    MmapInfo info = map_file(file, PROT_READ, MAP_PRIVATE | MAP_POPULATE);
    int64_t *data = (int64_t*)info.data;
    size_t count = info.bytes / sizeof(*data);
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < count; i++)
    {
        dest[i] = data[i];
    }
    unmap_file(&info);
}

Dataset* dataset_load(DatasetKind dskind, char const *root, SparseFormat format)
{
    Dataset *ds = ALLOC_OR_DIE(calloc(1, sizeof(*ds)));

    if (dskind <= DATASET_INVALID || dskind >= DATASET_COUNT) ERROR("Given dataset kind is not valid: %d", dskind);
    ds->info = &ds_infos[dskind];
    char *ds_path = path_join(root, ds->info->dir_name);
    if (access(ds_path, F_OK) != 0) ERROR("Dataset is missing, run:\n  dsprep --dataset %s --root %s", ds->info->name, root);

    char *label_bin_path = path_join(ds_path, "processed/node-label.bin");
    char *feat_bin_path  = path_join(ds_path, "processed/node-feat.bin");
    char *edge_bin_path  = path_join(ds_path, "processed/edge.bin");
    char *train_bin_path = path_join(ds_path, "processed/train.bin");
    char *valid_bin_path = path_join(ds_path, "processed/valid.bin");
    char *test_bin_path = path_join(ds_path, "processed/test.bin");
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
    ds->x             = ALLOC_OR_DIE(alloc_shared(node_count * feature_count * sizeof(*ds->x)));
    ds->y             = ALLOC_OR_DIE(alloc_local(node_count * sizeof(*ds->y)));
    ds->y_train       = ALLOC_OR_DIE(alloc_local(node_count * sizeof(*ds->y_train)));
    ds->y_valid       = ALLOC_OR_DIE(alloc_local(node_count * sizeof(*ds->y_valid)));
    ds->y_test        = ALLOC_OR_DIE(alloc_local(node_count * sizeof(*ds->y_test)));

    double label_time, feat_time, graph_time, split_time;
    TIMER_NORECORD(feat_time, load_feats(ds->feat_path, ds->x));
    TIMER_NORECORD(label_time, load_labels(ds->label_path, ds->y));
    TIMER_NORECORD(split_time, {
            create_split_mask(node_count, ds->y, train_bin_path, ds->y_train);
            create_split_mask(node_count, ds->y, valid_bin_path, ds->y_valid);
            create_split_mask(node_count, ds->y, test_bin_path, ds->y_test);
        });
    TIMER_NORECORD(graph_time,
                   ds->graph = sparsegraph_load(node_count, edge_count, ds->info->add_inverse_edge, edge_bin_path, format));
    printf("Loaded %s (nodes: %ld, edges: %ld) | Times: feat %.2fs, label %.2fs, split %.2fs, graph %.2fs\n",
           ds->info->name, ds->node_count, ds->edge_count,
           feat_time, label_time, split_time, graph_time);

    temp_free();
    return ds;
}

void dataset_free(Dataset **ds)
{
    if (!*ds) return;
    free((*ds)->label_path);
    free((*ds)->feat_path);
    free((*ds)->x);
    free((*ds)->y);
    free((*ds)->y_train);
    free((*ds)->y_valid);
    free((*ds)->y_test);
    sparsegraph_free(&(*ds)->graph);
    free(*ds);
    *ds = NULL;
}
