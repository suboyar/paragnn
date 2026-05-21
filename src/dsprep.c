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

static bool verbose = false;

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
        temp_buf = realloc(temp_buf, temp_cap);
        if (!temp_buf) { ERROR("out of memory"); exit(1); }
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

void parse_feats(char *input, size_t input_size, char *output, int64_t num_nodes, int64_t num_features)
{
    double *dest = (double *)output;
    char **line_starts = malloc((num_nodes + 1) * sizeof(char*));
    line_starts[0] = input;
    int64_t line = 1;
    for (char *p = input; p < input + input_size; p++)
    {
        if (*p == '\n' && line < num_nodes)
        {
            line_starts[line++] = p + 1;
        }
    }

#pragma omp parallel for
    for (int64_t i = 0; i < num_nodes; i++)
    {
        char *p = line_starts[i];
        for (int64_t j = 0; j < num_features; j++)
        {
            dest[i * num_features + j] = parse_double(&p);
            if (*p == ',') p++;
        }
    }
    free(line_starts);
}

void parse_labels(char *input, size_t input_size, char *output)
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

void parse_edges(char *input, size_t input_size, char *output)
{
    int64_t *dest = (int64_t *)output;
    char *p = input;
    char *end = input + input_size;
    size_t i = 0;
    while (p < end)
    {
        dest[2 * i]     = parse_i64(&p);
        if (*p == ',') p++;
        dest[2 * i + 1] = parse_i64(&p);
        if (*p == '\n') p++;
        i++;
    }
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

typedef enum { PARSE_FEATS, PARSE_LABELS, PARSE_EDGES, PARSE_SPLIT} ParseKind;
void process_csv_gz(const char *csv_gz_path, const char *bin_path, size_t out_size,
                    ParseKind kind, const DatasetInfo *ds_info)
{
    char *input = NULL;
    char *output = MAP_FAILED;
    int fd_out = -1;

    if (verbose) printf("Processing %s", bin_path);

    // Input
    size_t input_size = gz_decompress(csv_gz_path, (uint8_t**)&input);
    if (input == NULL) goto cleanup;

    if (kind == PARSE_SPLIT)
    {
        size_t count = 0;
        for (size_t i = 0; i < input_size; i++)
            if (input[i] == '\n') count++;
        // handle missing trailing newline
        if (input_size > 0 && input[input_size - 1] != '\n') count++;
        out_size = count * sizeof(int64_t);
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
    case PARSE_FEATS:  parse_feats(input, input_size, output, ds_info->num_nodes, ds_info->num_features); break;
    case PARSE_LABELS: parse_labels(input, input_size, output); break;
    case PARSE_EDGES:  parse_edges(input, input_size, output); break;
    case PARSE_SPLIT:  parse_splits(input, input_size, output); break;
    default: UNREACHABLE("Wrong parse kind: %d", kind);
    }

cleanup:
    free(input);
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

void process_npy(const char *npy_path, const char *bin_path, size_t out_size, size_t dst_elem_size)
{
    int rc = -1;
    int fd_in = -1;
    int fd_out = -1;
    char *input = MAP_FAILED;
    char *output = MAP_FAILED;
    struct stat sb = {0};

    printf("Processing %s", bin_path);

    fd_in = open(npy_path, O_RDONLY);
    if (fd_in < 0) ERROR("Could not open %s: %s", npy_path, strerror(errno));
    fstat(fd_in, &sb);
    input = mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd_in, 0);

    NpyHeader hdr = parse_npy_header(input);
    char *src = input + hdr.header_size;
    size_t total = out_size / dst_elem_size;

    fd_out = open(bin_path, O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd_out < 0)
    {
        ERROR("Could not open %s: %s", bin_path, strerror(errno));
    }
    ftruncate(fd_out, out_size);
    output = mmap(NULL, out_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_out, 0);

    // i8 -> u4
    if (hdr.type_char == 'i' && hdr.elem_size == 8 && dst_elem_size == 4)
    {
        int64_t  *s = (int64_t*)src;
        uint32_t *d = (uint32_t*)output;
        for (size_t i = 0; i < total; i++) d[i] = (uint32_t)s[i];
    }
    // i4 -> i8
    else if (hdr.type_char == 'i' && hdr.elem_size == 4 && dst_elem_size == 8)
    {
        int32_t *s = (int32_t*)src;
        int64_t *d = (int64_t*)output;
        for (size_t i = 0; i < total; i++) d[i] = (int64_t)s[i];
    }
    // f4 -> f8
    else if (hdr.type_char == 'f' && hdr.elem_size == 4 && dst_elem_size == 8)
    {
        float  *s = (float*)src;
        double *d = (double*)output;
        for (size_t i = 0; i < total; i++) d[i] = (double)s[i];
    }
    // f4 -> u4
    else if (hdr.type_char == 'f' && hdr.elem_size == 4 && dst_elem_size == 4)
    {
        float    *s = (float*)src;
        uint32_t *d = (uint32_t*)output;
        for (size_t i = 0; i < total; i++) d[i] = (uint32_t)s[i];
    }
    else if (hdr.elem_size == dst_elem_size)
    {
        memcpy(output, src, out_size);
    }
    else
    {
        ERROR("Unsupported npy conversion: %c%zu -> %zu",
                hdr.type_char, hdr.elem_size, dst_elem_size);
    }

    if (output != MAP_FAILED) munmap(output, out_size);
    if (input != MAP_FAILED)  munmap(input, sb.st_size);
    if (fd_out >= 0) close(fd_out);
    if (fd_in >= 0)  close(fd_in);
}

void prepare_dataset(char *dataset, char *datadir)
{
    const DatasetInfo *ds_info = &ds_infos[str_to_dataset_kind(dataset)];
    mkdir_recursive(datadir);

    const char *zip_name = path_name(ds_info->url);
    const char *zip_path = path_join(datadir, zip_name);
    const char *ds_path = path_join(datadir, ds_info->dir_name);
    const char *proc_path = path_join(ds_path, "processed");
    const char *split_root = path_join(ds_path, "split");
    const char *split_path  = path_join(split_root, ds_info->split_name);

    if (file_exists(path_join(proc_path, "edge.bin")) &&
        file_exists(path_join(proc_path, "node-feat.bin")) &&
        file_exists(path_join(proc_path, "node-label.bin")) &&
        file_exists(path_join(proc_path, "train.bin")) &&
        file_exists(path_join(proc_path, "valid.bin")) &&
        file_exists(path_join(proc_path, "test.bin")))
    {
        if (verbose) printf("Dataset %s already processed, skipping", ds_info->name);
        return;
    }

    if (!file_exists(path_join(ds_path, "raw")))
    {
        if (!file_exists(zip_path))
        {
            if (verbose) printf("Downloading %s...", ds_info->name);
            if (run_cmd((const char *const[]){"wget", "-q", "--show-progress", "-P", datadir, ds_info->url, NULL}) < 0)
            {
                ERROR("Failed to download %s dataset", ds_info->name);
            }
        }

        printf("Extracting %s...", ds_info->name);
        if (run_cmd((const char *const[]){"unzip", "-q", "-n", zip_path, "-d", datadir, NULL}) < 0)
        {
            ERROR("Failed to unzip %s dataset", ds_info->name);
        }
    }


    int64_t feat_size = ds_info->num_nodes * ds_info->num_features * sizeof(double);
    int64_t label_size = ds_info->num_nodes * sizeof(int64_t);
    int64_t edge_size = 2ULL * ds_info->num_edges * sizeof(int64_t);
    if (ds_info->raw_format == FMT_CSV_GZ)
    {
        // Data
        process_csv_gz(path_join(ds_path, "raw/node-feat.csv.gz"), path_join(proc_path, "node-feat.bin"), feat_size, PARSE_FEATS, ds_info);
        process_csv_gz(path_join(ds_path, "raw/node-label.csv.gz"), path_join(proc_path, "node-label.bin"), label_size, PARSE_LABELS, ds_info);
        process_csv_gz(path_join(ds_path, "raw/edge.csv.gz"), path_join(proc_path, "edge.bin"), edge_size, PARSE_EDGES, ds_info);

        // Splits
        process_csv_gz(path_join(split_path, "train.csv.gz"), path_join(proc_path, "train.bin"), 0, PARSE_SPLIT, ds_info);
        process_csv_gz(path_join(split_path, "valid.csv.gz"), path_join(proc_path, "valid.bin"), 0, PARSE_SPLIT, ds_info);
        process_csv_gz(path_join(split_path, "test.csv.gz"), path_join(proc_path, "test.bin"), 0, PARSE_SPLIT, ds_info);
    }
    else if (ds_info->raw_format == FMT_NPY)
    {
        int rc;
        rc = run_cmd((const char *const[]){"unzip", "-q",
                                           "-n", path_join(ds_path, "raw/data.npz"),
                                           "-d", path_join(ds_path, "raw/data")});
        if (rc < 0) ERROR("failed to unzip npz file: %s", path_join(ds_path, "raw/data.npz"));

        rc = run_cmd((const char *const[]){"unzip", "-q",
                                           "-n", path_join(ds_path, "raw/node-label.npz"),
                                           "-d", path_join(ds_path, "raw/node-label")});
        if (rc < 0) ERROR("failed to unzip npz file: %s", path_join(ds_path, "raw/node-label.npz"));

        process_npy(path_join(ds_path, "raw/data/node_feat.npy"), path_join(proc_path, "node-feat.bin"), feat_size, sizeof(double));
        process_npy(path_join(ds_path, "raw/node-label/node_label.npy"), path_join(proc_path, "node-label.bin"), label_size, sizeof(int64_t));
        process_npy(path_join(ds_path, "raw/data/edge_index.npy"), path_join(proc_path, "edge.bin"), edge_size, sizeof(int64_t));
    }
    else
    {
        ERROR("Invalid fromat: %d", ds_info->raw_format);
    }

    temp_reset();
}

static struct option long_options[] = {
    {"dataset", required_argument, NULL, 'd'},
    {"datadir", required_argument, NULL, 'D'},
    {"verbose", no_argument,       NULL, 'v'},
    {"help",    no_argument,       NULL, 'h'},
    {0,         0,                 0,     0}
};

void usage(const char *progname)
{
    fprintf(stderr,
            "Usage: %s -dataset NAME -datadir PATH\n"
            "  -d, -dataset  Dataset name {arxiv, products, papers100M}\n"
            "  -D, -datadir  Data directory\n"
            "  -v, -verbose  Verbose output\n",
            progname);
}

int main(int argc, char **argv)
{
    int rc = 1;

    char *dataset = NULL;
    char *datadir = NULL;

    int opt;
    while ((opt = getopt_long_only(argc, argv, "h", long_options, NULL)) != -1)
    {
        switch (opt)
        {
        case 'd': dataset = optarg; break;
        case 'D': datadir = optarg; break;
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

    if (!dataset || !datadir)
    {
        ERROR("both -dataset and -datadir are required");
        usage(argv[0]);
        goto exit;
    }

    datadir = expand_path(datadir);

    if (str_to_dataset_kind(dataset) == DATASET_INVALID)
    {
        ERROR("'%s' is an invalid dataset", dataset);
        usage(argv[0]);
        goto exit;
    }

    prepare_dataset(dataset, datadir);
    rc = 0;

exit:
    free(datadir);
    return rc;
}
