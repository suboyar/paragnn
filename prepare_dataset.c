#include <errno.h>
#include <omp.h>
#include <getopt.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <zlib.h>

#define NOB_IMPLEMENTATION
#include "nob.h"
#include "dataset_info.h"

Nob_Cmd cmd = {0};

static const char *path_join(const char *dir, const char *file)
{
    size_t dir_len = strlen(dir);
    if (dir[dir_len-1] != '/')
        return nob_temp_sprintf("%s/%s", dir, file);
    else
        return nob_temp_sprintf("%s%s", dir, file);
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
// then 32bit value. It does not do any space checking or clean-up.
static inline uint32_t parse_u32(char** pp)
{
    char* p = *pp;

    uint32_t val = 0;
    while (*p >= '0' && *p <= '9')
    {
        val = val * 10 + (*p++ - '0');
    }

    *pp = p;
    return val;
}

void parse_feats(char *input, size_t input_size, char *output, uint32_t num_nodes, uint32_t num_features)
{
    double *dest = (double *)output;
    char **line_starts = malloc((num_nodes + 1) * sizeof(char*));
    line_starts[0] = input;
    size_t line = 1;
    for (char *p = input; p < input + input_size; p++)
    {
        if (*p == '\n' && line < num_nodes)
        {
            line_starts[line++] = p + 1;
        }
    }

#pragma omp parallel for
    for (size_t i = 0; i < num_nodes; i++)
    {
        char *p = line_starts[i];
        for (size_t j = 0; j < num_features; j++)
        {
            dest[i * num_features + j] = parse_double(&p);
            if (*p == ',') p++;
        }
    }
    free(line_starts);
}

void parse_labels(char *input, size_t input_size, char *output)
{
    uint32_t *dest = (uint32_t *)output;
    char *p = input;
    char *end = input + input_size;
    size_t i = 0;
    while (p < end)
    {
        dest[i++] = parse_u32(&p);
        if (*p == '\n') p++;
    }
}

void parse_edges(char *input, size_t input_size, char *output)
{
    uint32_t *dest = (uint32_t *)output;
    char *p = input;
    char *end = input + input_size;
    size_t i = 0;
    while (p < end)
    {
        dest[2 * i]     = parse_u32(&p);
        if (*p == ',') p++;
        dest[2 * i + 1] = parse_u32(&p);
        if (*p == '\n') p++;
        i++;
    }
}

size_t gz_decompress(const char* file_path, char **buf)
{
    gzFile file = gzopen(file_path, "rb");
    if (!file)
    {
        fprintf(stderr, "gzopen failed: %s\n", strerror(errno));
        *buf = NULL;
        return 0;
    }

    size_t cap = 1 << 20;
    size_t len = 0;
    *buf = malloc(cap);


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
            *buf = realloc(*buf, cap);
        }
    }

    int err = 0;
    const char *error_string = "";
    int ret = gzeof(file);
    if (!ret)
    {
        error_string = gzerror(file, &err);
        fprintf(stderr, "gzread failed: %s\n", error_string);
        free(*buf);
        *buf = NULL;
        gzclose(file);
        return 0;
    }
    gzclose(file);
    return len;
}

void parse_splits(char *input, size_t input_size, char *output)
{
    uint32_t *dest = (uint32_t *)output;
    char *p = input;
    char *end = input + input_size;
    size_t i = 0;
    while (p < end)
    {
        dest[i++] = parse_u32(&p);
        if (*p == '\n') p++;
    }
}

typedef enum { PARSE_FEATS, PARSE_LABELS, PARSE_EDGES, PARSE_SPLIT} ParseKind;
bool process_csv_gz(const char *csv_gz_path, const char *bin_path, size_t out_size,
                    ParseKind kind, DatasetInfo *ds_info)
{
    nob_log(NOB_INFO, "Processing %s", bin_path);
    // Input
    char *input;
    size_t input_size = gz_decompress(csv_gz_path, &input);
    if (input == NULL) return false;

    if (kind == PARSE_SPLIT)
    {
        size_t count = 0;
        for (size_t i = 0; i < input_size; i++)
            if (input[i] == '\n') count++;
        // handle missing trailing newline
        if (input_size > 0 && input[input_size - 1] != '\n') count++;
        out_size = count * sizeof(uint32_t);
    }

    // Output
    int fd_out = open(bin_path, O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd_out < 0)
    {
        nob_log(NOB_ERROR, "Could not open %s: %s", csv_gz_path, strerror(errno));
        free(input);
        return false;
    }
    ftruncate(fd_out, out_size);
    char* output = mmap(NULL, out_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_out, 0);

    switch (kind) {
    case PARSE_FEATS:  parse_feats(input, input_size, output, ds_info->num_nodes, ds_info->num_features); break;
    case PARSE_LABELS: parse_labels(input, input_size, output); break;
    case PARSE_EDGES:  parse_edges(input, input_size, output); break;
    case PARSE_SPLIT:  parse_splits(input, input_size, output); break;
    default: NOB_UNREACHABLE(nob_temp_sprintf("Wrong parse kind: %d", kind));
    }

    free(input);
    munmap(output, out_size);
    close(fd_out);

    return true;
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

bool process_npy(const char *npy_path, const char *bin_path, size_t out_size, size_t dst_elem_size)
{
    bool ret = true;
    nob_log(NOB_INFO, "Processing %s", bin_path);

    int fd_in = open(npy_path, O_RDONLY);
    if (fd_in < 0)
    {
        nob_log(NOB_ERROR, "Could not open %s: %s", npy_path, strerror(errno));
        return false;
    }
    struct stat sb;
    fstat(fd_in, &sb);
    char *input = mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd_in, 0);

    NpyHeader hdr = parse_npy_header(input);
    char *src = input + hdr.header_size;
    size_t total = out_size / dst_elem_size;

    int fd_out = open(bin_path, O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd_out < 0)
    {
        nob_log(NOB_ERROR, "Could not open %s: %s", bin_path, strerror(errno));
        munmap(input, sb.st_size);
        close(fd_in);
        return false;
    }
    ftruncate(fd_out, out_size);
    char *output = mmap(NULL, out_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_out, 0);

    // f4 -> f8
    if (hdr.type_char == 'f' && hdr.elem_size == 4 && dst_elem_size == 8)
    {
        float *s = (float*)src; double *d = (double*)output;
        for (size_t i = 0; i < total; i++) d[i] = s[i];
    }
    // i8 -> u4
    else if (hdr.type_char == 'i' && hdr.elem_size == 8 && dst_elem_size == 4)
    {
        int64_t *s = (int64_t*)src; uint32_t *d = (uint32_t*)output;
        for (size_t i = 0; i < total; i++) d[i] = (uint32_t)s[i];
    }
    // f4 -> u4
    else if (hdr.type_char == 'f' && hdr.elem_size == 4 && dst_elem_size == 4)
    {
        float *s = (float*)src; uint32_t *d = (uint32_t*)output;
        for (size_t i = 0; i < total; i++) d[i] = (uint32_t)s[i];
    }
    else if (hdr.elem_size == dst_elem_size)
    {
        memcpy(output, src, out_size);
    }
    else
    {
        nob_log(NOB_ERROR, "Unsupported npy conversion: %c%zu -> %zu",
                hdr.type_char, hdr.elem_size, dst_elem_size);
        ret = false;
    }

    munmap(input, sb.st_size);
    close(fd_in);
    munmap(output, out_size);
    close(fd_out);
    return ret;
}

int prepare_dataset(char *dataset, char *datadir)
{
    DatasetInfo *ds_info = &ds_infos[str_to_dataset_kind(dataset)];
    if (!nob_mkdir_recursive(datadir)) return EXIT_FAILURE;

    const char *zip_name = nob_path_name(ds_info->url);
    const char *zip_path = nob_temp_sprintf("%s/%s", datadir, zip_name);
    const char *ds_path = path_join(datadir, ds_info->dir_name);
    const char *proc_path = path_join(ds_path, "processed");
    const char *split_root = path_join(ds_path, "split");
    const char *split_path  = path_join(split_root, ds_info->split_name);

    if (nob_file_exists(path_join(proc_path, "edge.bin")) &&
        nob_file_exists(path_join(proc_path, "node-feat.bin")) &&
        nob_file_exists(path_join(proc_path, "node-label.bin")) &&
        nob_file_exists(path_join(proc_path, "train.bin")) &&
        nob_file_exists(path_join(proc_path, "valid.bin")) &&
        nob_file_exists(path_join(proc_path, "test.bin")))
    {
        nob_log(NOB_INFO, "Dataset %s already processed, skipping", ds_info->name);
        return EXIT_SUCCESS;
    }

    if (!nob_file_exists(path_join(ds_path, "raw")))
    {
        if (!nob_file_exists(zip_path))
        {
            nob_log(NOB_INFO, "Downloading %s...", ds_info->name);
            nob_cmd_append(&cmd, "wget", "-q", "--show-progress", "-P", datadir, ds_info->url);
            if (!nob_cmd_run(&cmd))
            {
                nob_log(NOB_ERROR, "Failed to download %s dataset", ds_info->name);
                return EXIT_FAILURE;
            }
        }

        nob_log(NOB_INFO, "Extracting %s...", ds_info->name);
        nob_cmd_append(&cmd, "unzip", "-q", "-n", zip_path, "-d", datadir);
        if (!nob_cmd_run(&cmd))
        {
            nob_log(NOB_ERROR, "Failed to unzip %s dataset", ds_info->name);
            return EXIT_FAILURE;
        }
    }


    size_t feat_size = ds_info->num_nodes * ds_info->num_features * sizeof(double);
    size_t label_size = ds_info->num_nodes * sizeof(uint32_t);
    size_t edge_size = 2ULL * ds_info->num_edges * sizeof(uint32_t);
    if (ds_info->raw_format == FMT_CSV_GZ)
    {
        // Data
        if (!process_csv_gz(path_join(ds_path, "raw/node-feat.csv.gz"), path_join(proc_path, "node-feat.bin"), feat_size, PARSE_FEATS, ds_info))
            return EXIT_FAILURE;
        if (!process_csv_gz(path_join(ds_path, "raw/node-label.csv.gz"), path_join(proc_path, "node-label.bin"), label_size, PARSE_LABELS, ds_info))
            return EXIT_FAILURE;
        if (!process_csv_gz(path_join(ds_path, "raw/edge.csv.gz"), path_join(proc_path, "edge.bin"), edge_size, PARSE_EDGES, ds_info))
            return EXIT_FAILURE;

        // Splits
        if (!process_csv_gz(path_join(split_path, "train.csv.gz"), path_join(proc_path, "train.bin"), 0, PARSE_SPLIT, ds_info))
            return EXIT_FAILURE;
        if (!process_csv_gz(path_join(split_path, "valid.csv.gz"), path_join(proc_path, "valid.bin"), 0, PARSE_SPLIT, ds_info))
            return EXIT_FAILURE;
        if (!process_csv_gz(path_join(split_path, "test.csv.gz"), path_join(proc_path, "test.bin"), 0, PARSE_SPLIT, ds_info))
            return EXIT_FAILURE;
    }
    else if (ds_info->raw_format == FMT_NPY)
    {
        nob_cmd_append(&cmd, "unzip", "-q", "-n",
                       path_join(ds_path, "raw/data.npz"),
                       "-d",
                       path_join(ds_path, "raw/data"));
        if (!nob_cmd_run(&cmd)) return EXIT_FAILURE;

        nob_cmd_append(&cmd, "unzip", "-q", "-n",
                       nob_temp_sprintf("%s/raw/node-label.npz", ds_path),
                       "-d",
                       nob_temp_sprintf("%s/raw/node-label", ds_path));
        if (!nob_cmd_run(&cmd)) return EXIT_FAILURE;

        if (!process_npy(path_join(ds_path, "raw/data/node_feat.npy"), path_join(proc_path, "node-feat.bin"), feat_size, sizeof(double)))
            return EXIT_FAILURE;
        if (!process_npy(path_join(ds_path, "raw/node-label/node_label.npy"), path_join(proc_path, "node-label.bin"), label_size, sizeof(uint32_t)))
            return EXIT_FAILURE;
        if (!process_npy(nob_temp_sprintf(ds_path, "raw/data/edge_index.npy"), nob_temp_sprintf(proc_path, "edge.bin"), edge_size, sizeof(uint32_t)))
            return EXIT_FAILURE;
    }
    else
    {
        nob_log(NOB_ERROR, "Invalid fromat: %d", ds_info->raw_format);
        return EXIT_FAILURE;
    }

    nob_temp_reset();
    return EXIT_SUCCESS;
}

static struct option long_options[] = {
    {"dataset", required_argument, NULL, 'd'},
    {"datadir", required_argument, NULL, 'D'},
    {"help",    no_argument,       NULL, 'h'},
    {0,         0,                 0,     0}
};

void usage(const char *progname)
{
    fprintf(stderr,
            "Usage: %s -dataset NAME -datadir PATH\n"
            "  -dataset  Dataset name {arxiv, products, papers100M}\n"
            "  -datadir  Data directory\n",
            progname);
}

int main(int argc, char **argv)
{
    char *dataset = NULL;
    char *datadir = NULL;

    int opt;
    while ((opt = getopt_long_only(argc, argv, "h", long_options, NULL)) != -1)
    {
        switch (opt)
        {
        case 'd': dataset = optarg; break;
        case 'D': datadir = optarg; break;
        case 'h':
            usage(argv[0]);
            return 0;
        default:
            usage(argv[0]);
            return 1;
        }
    }

    if (!dataset || !datadir)
    {
        fprintf(stderr, "Error: both -dataset and -datadir are required\n");
        usage(argv[0]);
        return 1;
    }

    if (str_to_dataset_kind(dataset) == DATASET_INVALID)
    {
        fprintf(stderr, "Error: '%s' is an invalid dataset\n", dataset);
        usage(argv[0]);
        return 1;
    }

    printf("dataset: %s, datadir: %s\n", dataset, datadir);

    return prepare_dataset(dataset, datadir);
}
