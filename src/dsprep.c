#include <errno.h>
#include <fcntl.h>
#include <omp.h>
#include <getopt.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#include <zlib.h>

#include "core.h"
#include "dsinfo.h"

#define INVALID_IDX -1 // assumes signed node indecies
typedef enum {
    SPLIT_NONE = 0,
    SPLIT_TRAIN,
    SPLIT_VALID,
    SPLIT_TEST,
} Split;

typedef struct {
    int64_t total_nodes;
    int64_t total_edges;
    int64_t feature_count;
    bool add_inverse_edge;
} ParseContext;

#define CMD_ARGS(...) ((const char *const[]){__VA_ARGS__, NULL})

static bool verbose = false;

static char *temp_buf = NULL;
static size_t temp_cap = 0;
static size_t temp_used = 0;

typedef enum { PARSE_FEATS, PARSE_LABELS, PARSE_EDGES, PARSE_SPLIT} ParseKind;

static void temp_reset(void) { temp_used = 0; }
static void temp_free(void) { free(temp_buf); temp_buf = NULL; temp_cap = 0; temp_used = 0; }
static char *temp_alloc(size_t n)
{
    if (temp_used + n > temp_cap)
    {
        if (temp_cap == 0) temp_cap = 4096;
        while (temp_used + n > temp_cap) temp_cap *= 2;
        temp_buf = realloc(temp_buf, temp_cap);
        if (!temp_buf) ERROR("out of memory");
    }
    char *p = temp_buf + temp_used;
    temp_used += n;
    return p;
}

char *path_join(const char *dir, const char *file)
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

int run_cmd(const char *const argv[])
{
    pid_t pid = fork();
    if (pid < 0)
    {
        ERROR("fork failed: %s", strerror(errno));
    }

    if (pid == 0)
    {
        execvp(argv[0], (char *const *)argv);
        LOG_ERROR("execvp %s: %s", argv[0], strerror(errno));
        _exit(127);
    }

    int status;
    if (waitpid(pid, &status, 0) < 0)
    {
        ERROR("waitpid failed: %s\n", strerror(errno));
    }

    return WIFEXITED(status) && WEXITSTATUS(status) == 0 ? 0 : -1;
}

// This is specifically design to only conver cases for node-feat.csv with no space handling
static inline double parse_double(char** pp)
{
    char* p = *pp;
    double sign = 1.0;

    if (*p == '-') { sign = -1.0; p++; }
    else if (*p == '+') { p++; }

    int64_t intpart = 0;
    while (*p >= '0' && *p <= '9')
    {
        intpart = intpart * 10 + (*p++ - '0');
    }

    double val = (double)intpart;

    if (*p == '.')
    {
        p++;
        double scale = 0.1;
        while (*p >= '0' && *p <= '9')
        {
            val += (*p++ - '0') * scale;
            scale *= 0.1;
        }
    }

    if (*p == 'e' || *p == 'E')
    {
        p++;
        int exp_sign = 1;
        if (*p == '-') { exp_sign = -1; p++; }
        else if (*p == '+') { p++; }

        int exp = 0;
        while (*p >= '0' && *p <= '9')
        {
            exp = exp * 10 + (*p++ - '0');
        }

        if (exp_sign > 0)
        {
            while (exp-- > 0) val *= 10.0;
        }
        else
        {
            while (exp-- > 0) val *= 0.1;
        }
    }

    *pp = p;
    return sign * val;
}

// This is specifically design to only be used for node indicies that aren't bigger
// then signed 64bit value. It does not do any space checking or clean-up.
static inline int64_t parse_i64(char** pp)
{
    char* p = *pp;

    int64_t val = 0;
    while (*p >= '0' && *p <= '9')
    {
        val = val * 10 + (*p++ - '0');
    }

    *pp = p;
    return val;
}

void parse_feats(char *input, size_t input_size, char *output, int64_t total_nodes, int64_t split_size, int64_t feature_count, int64_t *split_idx)
{
    double *dest = (double *)output;
    char **line_starts = malloc((total_nodes + 1) * sizeof(char*));
    line_starts[0] = input;
    int64_t line = 1;
    for (char *p = input; p < input + input_size; p++)
    {
        if (*p == '\n' && line < total_nodes) line_starts[line++] = p + 1;
    }

    if (split_idx == NULL)
    {
#pragma omp parallel for
        for (int64_t i = 0; i < total_nodes; i++)
        {
            char *p = line_starts[i];
            for (int64_t j = 0; j < feature_count; j++)
            {
                dest[i * feature_count + j] = parse_double(&p);
                if (*p == ',') p++;
            }
        }
    }
    else
    {
#pragma omp parallel for
        for (int64_t i = 0; i < split_size; i++)
        {
            char *p = line_starts[split_idx[i]];
            for (int64_t j = 0; j < feature_count; j++)
            {
                dest[i * feature_count + j] = parse_double(&p);
                if (*p == ',') p++;
            }
        }
    }
    free(line_starts);
}

void parse_labels(char *input, size_t input_size, char *output, int64_t total_nodes, int64_t split_size, int64_t *split_idx)
{
    int64_t *dest = (int64_t *)output;
    char **line_starts = malloc((total_nodes + 1) * sizeof(char*));
    line_starts[0] = input;
    int64_t line = 1;
    for (char *p = input; p < input + input_size; p++)
    {
        if (*p == '\n' && line < total_nodes) line_starts[line++] = p + 1;
    }

    if (split_idx == NULL)
    {
#pragma omp parallel for
        for (int64_t i = 0; i < total_nodes; i++)
        {
            char *p = line_starts[i];
            dest[i] = parse_i64(&p);
        }
    }
    else
    {
#pragma omp parallel for
        for (int64_t i = 0; i < split_size; i++)
        {
            char *p = line_starts[split_idx[i]];
            dest[i] = parse_i64(&p);

        }
    }
}

int cmp_s128(const void *a, const void *b) {
    signed __int128 x = *(const signed __int128 *)a;
    signed __int128 y = *(const signed __int128 *)b;
    return (x > y) - (x < y);
}

void parse_edges(char *input, size_t input_size, char *output, size_t edge_count, bool symmetrize, int fd_out, int64_t *node_map)
{
    signed __int128 *packed = malloc(edge_count * sizeof(*packed));
    char *p = input;
    char *end = input + input_size;
    size_t n = 0;

    // TODO: parallelize this
    if (node_map == NULL)
    {
        while (p < end)
        {
            int64_t u = parse_i64(&p);
            if (*p == ',') p++;
            int64_t v = parse_i64(&p);
            if (*p == '\n') p++;

            packed[n++] = ((signed __int128)u << 64) | (uint64_t)v;
            if (symmetrize && u != v)
            {
                packed[n++] = ((signed __int128)v << 64) | (uint64_t)u;
            }
        }
    }
    else
    {
        while (p < end)
        {
            int64_t u = node_map[parse_i64(&p)];
            if (*p == ',') p++;
            int64_t v = node_map[parse_i64(&p)];
            if (*p == '\n') p++;

            if (u == INVALID_IDX || v == INVALID_IDX) continue;

            packed[n++] = ((signed __int128)u << 64) | (uint64_t)v;
            if (symmetrize && u != v)
            {
                packed[n++] = ((signed __int128)v << 64) | (uint64_t)u;
            }
        }
    }

    qsort(packed, n, sizeof(*packed), cmp_s128);

    size_t unique = 0;
    if (n > 0)
    {
        unique = 1;
        for (size_t i = 1; i < n; i++)
        {
            if (packed[i] != packed[unique - 1])
                packed[unique++] = packed[i];
        }
    }


    int64_t *dest = (int64_t *)output;
#pragma omp parallel for
    for (size_t i = 0; i < unique; i++)
    {
#if INTERLEAVED_EDGES // src0, dst0, src1, dst1
        dest[2 * i]     = (int64_t)(packed[i] >> 64);
        dest[2 * i + 1] = (int64_t)packed[i];
#else // src0,...,dst0,...
        dest[i] = (int64_t)(packed[i] >> 64);
        dest[i + unique] = (int64_t)packed[i];
#endif
    }

    free(packed);
    ftruncate(fd_out, unique * 2 * sizeof(int64_t));
}

size_t gz_decompress(const char* file_path, uint8_t **buf)
{
    gzFile file = NULL;
    size_t len = 0;
    size_t cap = 1 << 20;
    int success = 0;

    *buf = NULL;

    if (!(file = gzopen(file_path, "rb")))
    {
        ERROR("gzopen failed: %s", strerror(errno));
        goto cleanup;
    }

    *buf = malloc(cap);
    if (!*buf)
    {
        ERROR("malloc failed: %s", strerror(errno));
    }

    while (1)
    {
        size_t remaining = cap - len;
        unsigned int chunk = remaining > INT_MAX ? INT_MAX : (unsigned int)remaining;
        int n = gzread(file, *buf + len, chunk);
        if (n <= 0) break;
        len += n;
        if (len == cap)
        {
            cap *= 2;
            uint8_t *tmp = realloc(*buf, cap);
            if (!tmp)
            {
                ERROR("realloc failed: %s", strerror(errno));
            }
            *buf = tmp;
        }
    }

    if (!gzeof(file))
    {
        int err;
        const char *msg = gzerror(file, &err);
        ERROR("gzread failed: %s", msg);
    }

    success = 1;

cleanup:
    if (file) gzclose(file);
    if (!success)
    {
        free(*buf);
        *buf = NULL;
        len = 0;
    }
    return len;
}

void parse_splits(char *input, size_t input_size, char *output)
{
    int64_t *dest = (int64_t *)output;
    char *p = input;
    char *end = input + input_size;
    size_t i = 0;
    while (p < end)
    {
        dest[i++] = parse_i64(&p);
        if (*p == '\n') p++;
    }
}

int64_t load_split(const char *path, int64_t **split)
{
    int fd = open(path, O_RDONLY);
    if (fd < 0) ERROR("Could not open %s: %s", path, strerror(errno));
    struct stat sb;
    fstat(fd, &sb);
    int64_t* data = mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (data == MAP_FAILED) ERROR("mmap failed: %s", strerror(errno));
    *split = malloc(sb.st_size);
    size_t count = sb.st_size / sizeof(*data);
#pragma omp parallel for
    for (size_t i = 0; i < count; i++)
    {
        (*split)[i] = data[i];
    }
    munmap(data, sb.st_size);
    close(fd);
    return count;
}

void build_node_mapping(int64_t* map,  int64_t num_nodes, int64_t* split_idx, int64_t split_size)
{
    memset(map, INVALID_IDX, num_nodes * sizeof(*map));

    for (int64_t i = 0; i < split_size; i++)
    {
        map[split_idx[i]] = i;
    }
}

void process_csv_gz(const char *csv_gz_path, const char *bin_path, const char *split_path, ParseKind kind, Split split, const ParseContext *ctx)
{
    char *input = NULL;
    char *output = MAP_FAILED;
    int fd_out = -1;
    size_t out_size;

    if (file_exists(bin_path))
    {
        printf("Skipping %s\n", bin_path);
        return;
    }
    printf("Processing %s\n", bin_path);

    // Input
    size_t input_size = gz_decompress(csv_gz_path, (uint8_t**)&input);
    if (input == NULL) goto cleanup;

    int64_t *node_map = NULL, *split_idx = NULL;
    int64_t split_nodes = (split == SPLIT_NONE) ? ctx->total_nodes : load_split(split_path, &split_idx);
    int64_t edge_count = ctx->total_edges;
    if (kind == PARSE_EDGES && split != SPLIT_NONE)
    {
        node_map = malloc(ctx->total_nodes * sizeof(*node_map));
        build_node_mapping(node_map, ctx->total_nodes, split_idx, split_nodes);

        edge_count = 0;
        char *p = input, *end = input + input_size;

        while (p < end)
        {
            int64_t u = node_map[parse_i64(&p)];
            if (*p == ',') p++;
            int64_t v = node_map[parse_i64(&p)];
            if (*p == '\n') p++;

            if (u != INVALID_IDX && v != INVALID_IDX) edge_count++;
        }
    }

    edge_count *= (ctx->add_inverse_edge ? 2 : 1);

    switch (kind) {
        case PARSE_FEATS: out_size = split_nodes * ctx->feature_count * sizeof(double); break;
        case PARSE_LABELS: out_size = split_nodes * sizeof(int64_t); break;
        case PARSE_EDGES: out_size = 2 * edge_count * sizeof(int64_t); break;
        case PARSE_SPLIT:
        {
            size_t count = 0;
            for (size_t i = 0; i < input_size; i++)
                if (input[i] == '\n') count++;
            // Handle missing trailing newline
            if (input_size > 0 && input[input_size - 1] != '\n') count++;
            out_size = count * sizeof(int64_t);
            break;
        }
        default: UNREACHABLE("Wrong parse kind: %d", kind);
    }

    // Output
    fd_out = open(bin_path, O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd_out < 0)
    {
        ERROR("Could not open %s: %s", csv_gz_path, strerror(errno));
        goto cleanup;
    }
    ftruncate(fd_out, out_size);
    output = mmap(NULL, out_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_out, 0);

    switch (kind) {
        case PARSE_FEATS: parse_feats(input, input_size, output, ctx->total_nodes, split_nodes, ctx->feature_count, split_idx); break;
        case PARSE_LABELS: parse_labels(input, input_size, output, ctx->total_nodes, split_nodes, split_idx); break;
        case PARSE_EDGES: parse_edges(input, input_size, output, edge_count, ctx->add_inverse_edge, fd_out, node_map); break;
        case PARSE_SPLIT:  parse_splits(input, input_size, output); break;
        default: UNREACHABLE("Wrong parse kind: %d", kind);
    }

cleanup:
    free(input);
    free(node_map);
    free(split_idx);
    if (output != MAP_FAILED) munmap(output, out_size);
    if (fd_out >= 0) close(fd_out);
}

typedef struct {
    size_t header_size;
    size_t elem_size;
    char   type_char;
} NpyHeader;

NpyHeader parse_npy_header(const char *data)
{
    NpyHeader h = {0};
    uint8_t major = data[6];
    uint16_t len2; uint32_t len4;

    if (major == 1) { memcpy(&len2, data+8, 2); h.header_size = 10 + len2; }
    else            { memcpy(&len4, data+8, 4); h.header_size = 12 + len4; }

    const char *q = strstr(data + (major == 1 ? 10 : 12), "'descr'");
    if (q) { q = strchr(q+7, '\'') + 1; h.type_char = q[1]; h.elem_size = q[2] - '0'; }

    return h;
}


void process_npy(const char *npy_path, const char *bin_path, const char *split_path,
                 ParseKind kind, Split split, const ParseContext *ctx)
{
    int fd_in = -1, fd_out = -1;
    char *input = MAP_FAILED, *output = MAP_FAILED;
    struct stat sb = {0};

    if (file_exists(bin_path))
    {
        printf("Skipping %s\n", bin_path);
        return;
    }
    printf("Processing %s\n", bin_path);

    fd_in = open(npy_path, O_RDONLY);
    if (fd_in < 0) ERROR("Could not open %s: %s", npy_path, strerror(errno));
    fstat(fd_in, &sb);
    input = mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd_in, 0);
    if (input == MAP_FAILED) ERROR("mmap input failed for %s: %s", npy_path, strerror(errno));

    NpyHeader hdr = parse_npy_header(input);
    char *src = input + hdr.header_size;

    int64_t *node_map = NULL, *split_idx = NULL;
    int64_t split_nodes = (split == SPLIT_NONE) ? ctx->total_nodes : load_split(split_path, &split_idx);

    if (kind == PARSE_EDGES)
    {
        size_t alloc_edges = ctx->add_inverse_edge ? 2 * ctx->total_edges : ctx->total_edges;
        signed __int128 *packed = malloc(alloc_edges * sizeof(*packed));
        size_t n = 0;

        if (split != SPLIT_NONE)
        {
            node_map = malloc(ctx->total_nodes * sizeof(*node_map));
            build_node_mapping(node_map, ctx->total_nodes, split_idx, split_nodes);
        }

#define EXTRACT_EDGES(stype) do {                                       \
            const stype *s = (const stype *)src;                        \
            if (!ctx->add_inverse_edge && !node_map) {                  \
                _Pragma("omp parallel for")                             \
                    for (size_t i = 0; i < ctx->total_edges; i++) {       \
                        int64_t u = s[i], v = s[ctx->total_edges + i];    \
                        packed[i] = ((signed __int128)u << 64) | (uint64_t)v; \
                    }                                                   \
                n = ctx->total_edges;                                     \
            } else {                                                    \
                for (size_t i = 0; i < ctx->total_edges; i++) {           \
                    int64_t u = s[i], v = s[ctx->total_edges + i];        \
                    if (node_map) {                                     \
                        u = node_map[u]; v = node_map[v];               \
                        if (u == INVALID_IDX || v == INVALID_IDX) continue; \
                    }                                                   \
                    packed[n++] = ((signed __int128)u << 64) | (uint64_t)v; \
                    if (ctx->add_inverse_edge && u != v) {              \
                        packed[n++] = ((signed __int128)v << 64) | (uint64_t)u; \
                    }                                                   \
                }                                                       \
            }                                                           \
        } while(0)

        // ogbn-papers100M seems to be stored as src0,...,srcE,dst0,...,dstE
        if (hdr.elem_size == 4) EXTRACT_EDGES(int32_t); // font-lock-function-name-face
        else EXTRACT_EDGES(int64_t); // no font-lock-function-name-face

        qsort(packed, n, sizeof(*packed), cmp_s128); //font-lock-function-name-face

        size_t unique = 0;
        if (n > 0)
        {
            unique = 1;
            for (size_t i = 1; i < n; i++)
            {
                if (packed[i] != packed[unique - 1])
                    packed[unique++] = packed[i];
            }
        }

        size_t out_size = unique * 2 * sizeof(int64_t);
        fd_out = open(bin_path, O_RDWR | O_CREAT | O_TRUNC, 0644);
        ftruncate(fd_out, out_size); // font-lock-function-name-face
        output = mmap(NULL, out_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_out, 0);

        int64_t *dest = (int64_t *)output;
#pragma omp parallel for
        for (size_t i = 0; i < unique; i++)
        {
#if INTERLEAVED_EDGES // src0, dst0, src1, dst1
            dest[2 * i]     = (int64_t)(packed[i] >> 64);
            dest[2 * i + 1] = (int64_t)packed[i];
#else // src0,...,dst0,...
            dest[i] = (int64_t)(packed[i] >> 64);
            dest[i + unique] = (int64_t)packed[i];
#endif
        }

        free(packed); // font-lock-function-name-face
        if (node_map) free(node_map); // font-lock-function-name-face
        munmap(output, out_size); //font-lock-function-name-face
        close(fd_out); // font-lock-function-name-face
    }
    else
    {
        size_t dst_elem_size = (kind == PARSE_FEATS) ? sizeof(double) : sizeof(int64_t);
        size_t cols = (kind == PARSE_FEATS) ? ctx->feature_count : 1;
        size_t out_size = split_nodes * cols * dst_elem_size;

        fd_out = open(bin_path, O_RDWR | O_CREAT | O_TRUNC, 0644); // no font-lock-function-name-face
        ftruncate(fd_out, out_size); // no font-lock-function-name-face
        output = mmap(NULL, out_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_out, 0); // no font-lock-function-name-face
        if (output == MAP_FAILED) ERROR("mmap output failed: %s", strerror(errno))
;
        #define COPY_NPY(stype, dtype) do { \
            stype *s = (stype*)src; dtype *d = (dtype*)output; \
            _Pragma("omp parallel for") \
            for (size_t i = 0; i < split_nodes; i++) { \
                size_t row = split_idx ? split_idx[i] : i; \
                for (size_t j = 0; j < cols; j++) \
                    d[i * cols + j] = (dtype)s[row * cols + j]; \
            } \
        } while(0)

        if (hdr.type_char == 'i' && hdr.elem_size == 8 && dst_elem_size == 4) COPY_NPY(int64_t, uint32_t);
        else if (hdr.type_char == 'i' && hdr.elem_size == 4 && dst_elem_size == 8) COPY_NPY(int32_t, int64_t);
        else if (hdr.type_char == 'f' && hdr.elem_size == 4 && dst_elem_size == 8) COPY_NPY(float, double);
        else if (hdr.type_char == 'f' && hdr.elem_size == 4 && dst_elem_size == 4) COPY_NPY(float, uint32_t);
        else if (hdr.elem_size == dst_elem_size)
        {
            if (!split_idx) memcpy(output, src, out_size);
            else if (hdr.elem_size == 8) COPY_NPY(int64_t, int64_t);
            else if (hdr.elem_size == 4) COPY_NPY(int32_t, int32_t);
        }
        else ERROR("Unsupported npy conversion: %c%zu -> %zu", hdr.type_char, hdr.elem_size, dst_elem_size);

        munmap(output, out_size);
        close(fd_out);
    }

cleanup:
    if (split_idx) free(split_idx);
    munmap(input, sb.st_size);
    close(fd_in);
}

int64_t count_lines_gz(const char *path)
{
    uint8_t *buf = NULL;
    size_t size = gz_decompress(path, &buf);
    if (!buf) ERROR("Failed to decompress %s for line counting", path);

    int64_t count = 0;
#pragma omp parallel for reduction(+:count)
    for (size_t i = 0; i < size; i++)
    {
        if (buf[i] == '\n') count++;
    }
    if (size > 0 && buf[size - 1] != '\n') count++;

    free(buf);
    return count;
}

int64_t count_npy_elements(const char *path)
{
    int fd = open(path, O_RDONLY);
    if (fd < 0) ERROR("Could not open %s: %s", path, strerror(errno));

    struct stat sb;
    fstat(fd, &sb);
    char *data = mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (data == MAP_FAILED) ERROR("mmap failed for %s", path);

    NpyHeader hdr = parse_npy_header(data);
    int64_t count = (sb.st_size - hdr.header_size) / hdr.elem_size;

    munmap(data, sb.st_size);
    close(fd);
    return count;
}

void prepare_dataset(DatasetKind kind, char *root)
{
    const DatasetInfo *ds_info = &ds_infos[kind];
    mkdir_recursive(root);

    const char *zip_name = path_name(ds_info->url);
    const char *zip_path = path_join(root, zip_name);
    const char *ds_path = path_join(root, ds_info->dir_name);
    const char *proc_path = path_join(ds_path, "processed");
    const char *split_root = path_join(ds_path, "split");
    const char *split_path  = path_join(split_root, ds_info->split);

    if (!file_exists(path_join(ds_path, "raw")))
    {
        if (!file_exists(zip_path))
        {
            printf("Downloading %s...", ds_info->name);
            const char *const *args = verbose
                ? CMD_ARGS("wget", "-q", "--show-progress", "-P", root, ds_info->url)
                : CMD_ARGS("wget", "--show-progress", "-P", root, ds_info->url);
            if (run_cmd(args) < 0) ERROR("Failed to download %s dataset", ds_info->name);
        }

        printf("Extracting %s...", ds_info->name);

        const char *const *args = verbose
            ? CMD_ARGS("unzip", "-q", "-n", zip_path, "-d", root)
            : CMD_ARGS("unzip", "-n", zip_path, "-d", root);
        if (run_cmd(args) < 0) ERROR("Failed to unzip %s dataset", ds_info->name);

        const char *extracted_path = path_join(root, ds_info->download_name);
        if (strcmp(extracted_path, ds_path) != 0)
        {
            if (rename(extracted_path, ds_path) < 0)
            {
                ERROR("Failed to rename %s to %s: %s", extracted_path, ds_path, strerror(errno));
            }
        }
    }

    int64_t total_nodes = 0;
    int64_t total_edges = 0;

    if (ds_info->raw_format == FMT_CSV_GZ)
    {
        total_nodes = count_lines_gz(path_join(ds_path, "raw/node-label.csv.gz"));
        total_edges = count_lines_gz(path_join(ds_path, "raw/edge.csv.gz"));
    }
    else if (ds_info->raw_format == FMT_NPY)
    {
        total_nodes = count_npy_elements(path_join(ds_path, "raw/node-label/node_label.npy"));
        // Edges array is 2D (source, dest), so its divide by 2
        total_edges = count_npy_elements(path_join(ds_path, "raw/data/edge_index.npy")) / 2;
    }

    ParseContext ctx = {
        .total_nodes      = total_nodes,
        .total_edges      = total_edges,
        .feature_count    = ds_info->feature_count,
        .add_inverse_edge = ds_info->add_inverse_edge
    };

    // Splits, they seem to be only *.csv_gz formated regardles for ds_info->raw_format
    process_csv_gz(path_join(split_path, "train.csv.gz"), path_join(proc_path, "train.bin"), NULL, PARSE_SPLIT, SPLIT_NONE, &ctx);
    process_csv_gz(path_join(split_path, "valid.csv.gz"), path_join(proc_path, "valid.bin"), NULL, PARSE_SPLIT, SPLIT_NONE, &ctx);
    process_csv_gz(path_join(split_path, "test.csv.gz"),  path_join(proc_path, "test.bin"),  NULL, PARSE_SPLIT, SPLIT_NONE, &ctx);

    Split splits[] = {SPLIT_NONE, SPLIT_TRAIN, SPLIT_VALID, SPLIT_TEST};

    // Data
    if (ds_info->raw_format == FMT_CSV_GZ)
    {
        // Full
        process_csv_gz(path_join(ds_path, "raw/node-feat.csv.gz"),  path_join(proc_path, "node-feat.bin"),  NULL, PARSE_FEATS,  SPLIT_NONE, &ctx);
        process_csv_gz(path_join(ds_path, "raw/node-label.csv.gz"), path_join(proc_path, "node-label.bin"), NULL, PARSE_LABELS, SPLIT_NONE, &ctx);
        process_csv_gz(path_join(ds_path, "raw/edge.csv.gz"),       path_join(proc_path, "edge.bin"),       NULL, PARSE_EDGES,  SPLIT_NONE, &ctx);

        // Training split
        process_csv_gz(path_join(ds_path, "raw/node-feat.csv.gz"),  path_join(proc_path, "train-node-feat.bin"),  path_join(proc_path, "train.bin"), PARSE_FEATS,  SPLIT_TRAIN, &ctx);
        process_csv_gz(path_join(ds_path, "raw/node-label.csv.gz"), path_join(proc_path, "train-node-label.bin"), path_join(proc_path, "train.bin"), PARSE_LABELS, SPLIT_TRAIN, &ctx);
        process_csv_gz(path_join(ds_path, "raw/edge.csv.gz"),       path_join(proc_path, "train-edge.bin"),       path_join(proc_path, "train.bin"), PARSE_EDGES,  SPLIT_TRAIN, &ctx);

        // Validation split
        process_csv_gz(path_join(ds_path, "raw/node-feat.csv.gz"),  path_join(proc_path, "valid-node-feat.bin"),  path_join(proc_path, "valid.bin"), PARSE_FEATS,  SPLIT_VALID, &ctx);
        process_csv_gz(path_join(ds_path, "raw/node-label.csv.gz"), path_join(proc_path, "valid-node-label.bin"), path_join(proc_path, "valid.bin"), PARSE_LABELS, SPLIT_VALID, &ctx);
        process_csv_gz(path_join(ds_path, "raw/edge.csv.gz"),       path_join(proc_path, "valid-edge.bin"),       path_join(proc_path, "valid.bin"), PARSE_EDGES,  SPLIT_VALID, &ctx);

        // Test split
        process_csv_gz(path_join(ds_path, "raw/node-feat.csv.gz"),  path_join(proc_path, "test-node-feat.bin"),  path_join(proc_path, "test.bin"), PARSE_FEATS,  SPLIT_TEST, &ctx);
        process_csv_gz(path_join(ds_path, "raw/node-label.csv.gz"), path_join(proc_path, "test-node-label.bin"), path_join(proc_path, "test.bin"), PARSE_LABELS, SPLIT_TEST, &ctx);
        process_csv_gz(path_join(ds_path, "raw/edge.csv.gz"),       path_join(proc_path, "test-edge.bin"),       path_join(proc_path, "test.bin"), PARSE_EDGES,  SPLIT_TEST, &ctx);
    }
    else if (ds_info->raw_format == FMT_NPY)
    {
        int rc;
        rc = run_cmd(CMD_ARGS("unzip", "-q",
                              "-n", path_join(ds_path, "raw/data.npz"),
                              "-d", path_join(ds_path, "raw/data")));
        if (rc < 0) ERROR("failed to unzip npz file: %s", path_join(ds_path, "raw/data.npz"));

        rc = run_cmd(CMD_ARGS("unzip", "-q",
                              "-n", path_join(ds_path, "raw/node-label.npz"),
                              "-d", path_join(ds_path, "raw/node-label")));
        if (rc < 0) ERROR("failed to unzip npz file: %s", path_join(ds_path, "raw/node-label.npz"));

        // Full
        process_npy(path_join(ds_path, "raw/data/node_feat.npy"),        path_join(proc_path, "node-feat.bin"),  NULL, PARSE_FEATS,  SPLIT_NONE, &ctx);
        process_npy(path_join(ds_path, "raw/node-label/node_label.npy"), path_join(proc_path, "node-label.bin"), NULL, PARSE_LABELS, SPLIT_NONE, &ctx);
        process_npy(path_join(ds_path, "raw/data/edge_index.npy"),       path_join(proc_path, "edge.bin"),       NULL, PARSE_EDGES,  SPLIT_NONE, &ctx);

        // Training split
        process_npy(path_join(ds_path, "raw/data/node_feat.npy"),        path_join(proc_path, "train-node-feat.bin"),  path_join(proc_path, "train.bin"), PARSE_FEATS,  SPLIT_TRAIN, &ctx);
        process_npy(path_join(ds_path, "raw/node-label/node_label.npy"), path_join(proc_path, "train-node-label.bin"), path_join(proc_path, "train.bin"), PARSE_LABELS, SPLIT_TRAIN, &ctx);
        process_npy(path_join(ds_path, "raw/data/edge_index.npy"),       path_join(proc_path, "train-edge.bin"),       path_join(proc_path, "train.bin"), PARSE_EDGES,  SPLIT_TRAIN, &ctx);

        // Validation split
        process_npy(path_join(ds_path, "raw/data/node_feat.npy"),        path_join(proc_path, "valid-node-feat.bin"),  path_join(proc_path, "valid.bin"), PARSE_FEATS,  SPLIT_VALID, &ctx);
        process_npy(path_join(ds_path, "raw/node-label/node_label.npy"), path_join(proc_path, "valid-node-label.bin"), path_join(proc_path, "valid.bin"), PARSE_LABELS, SPLIT_VALID, &ctx);
        process_npy(path_join(ds_path, "raw/data/edge_index.npy"),       path_join(proc_path, "valid-edge.bin"),       path_join(proc_path, "valid.bin"), PARSE_EDGES,  SPLIT_VALID, &ctx);

        // Test split
        process_npy(path_join(ds_path, "raw/data/node_feat.npy"),        path_join(proc_path, "test-node-feat.bin"),  path_join(proc_path, "test.bin"), PARSE_FEATS,  SPLIT_TEST, &ctx);
        process_npy(path_join(ds_path, "raw/node-label/node_label.npy"), path_join(proc_path, "test-node-label.bin"), path_join(proc_path, "test.bin"), PARSE_LABELS, SPLIT_TEST, &ctx);
        process_npy(path_join(ds_path, "raw/data/edge_index.npy"),       path_join(proc_path, "test-edge.bin"),       path_join(proc_path, "test.bin"), PARSE_EDGES,  SPLIT_TEST, &ctx);
    }
    else ERROR("Invalid fromat: %d", ds_info->raw_format);

    temp_reset();
}

static struct option long_options[] = {
    {"dataset", required_argument, NULL, 'd'},
    {"root",    required_argument, NULL, 'r'},
    {"verbose", no_argument,       NULL, 'v'},
    {"help",    no_argument,       NULL, 'h'},
    {0,         0,                 0,     0}
};

void usage(const char *progname)
{
    fprintf(stderr,
            "Usage: %s -d NAME -r PATH\n"
            "  -d, --dataset  Dataset name {ogbn-arxiv, ogbn-products, ogbn-papers100M, all}\n"
            "  -r, --root     Data directory root path\n"
            "  -v, --verbose  Verbose output\n"
            "  -h, --help     Show help\n",
            progname);
}

int main(int argc, char **argv)
{
    int rc = 1;

    char *dataset = NULL;
    char *root = NULL;

    int opt;
    while ((opt = getopt_long(argc, argv, "d:r:vh", long_options, NULL)) != -1)
    {
        switch (opt)
        {
            case 'd': dataset = optarg; break;
            case 'r': root = optarg; break;
            case 'v': verbose = true; break;
            case 'h':
                usage(argv[0]);
                rc = 0;
                goto exit;
            default:
                usage(argv[0]);
                goto exit;
        }
    }

    if (!dataset || !root)
    {
        fprintf(stderr, "ERROR: Both -dataset and -datadir are required\n");
        usage(argv[0]);
        goto exit;
    }

    root = expand_path(root);

    if (strcmp(dataset, "all") == 0)
    {
        DatasetKind all_datasets[] = {DATASET_PRODUCTS, DATASET_ARXIV, DATASET_PAPERS100M};
        size_t count = sizeof(all_datasets) / sizeof(all_datasets[0]);

        for (size_t i = 0; i < count; i++)
        {
            prepare_dataset(all_datasets[i], root);
        }
    }
    else
    {
        DatasetKind kind = str_to_dataset_kind(dataset);
        if (kind == DATASET_INVALID)
        {
            fprintf(stderr, "ERROR: '%s' is an invalid dataset\n", dataset);
            usage(argv[0]);
            goto exit;
        }
        prepare_dataset(kind, root);
    }
    rc = 0;

exit:
    free(root);
    return rc;
}
