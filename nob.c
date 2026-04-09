#include <sys/mman.h>
#include <sys/stat.h>
#include <zlib.h>

#define NOB_EXPERIMENTAL_DELETE_OLD
#define NOB_IMPLEMENTATION
#define NOB_WARN_DEPRECATED
#define NOB_REBUILD_URSELF(binary_path, source_path) \
    "gcc", "-ggdb", "-fopenmp", "-o", binary_path, source_path, "-lz"
#define nob_cc(cmd) nob_cmd_append(cmd, "gcc")
#include "nob.h"
#undef nob_cc_flags

#define nob_cc_flags(cmd) nob_cmd_append(cmd, "-std=c17", "-D_POSIX_C_SOURCE=200809L")
#define nob_cc_error_flags(cmd) \
    nob_cmd_append(cmd,                                                 \
                   "-Wall",                                             \
                   "-Wextra",                                           \
                   "-Wfloat-conversion",                                \
                   "-Werror=implicit-function-declaration",             \
                   "-Werror=strict-prototypes",                         \
                   "-Werror=incompatible-pointer-types") // Maybe re-add -Wno-unknown-pragmas?

#define FLAG_IMPLEMENTATION
#define FLAG_PUSH_DASH_DASH_BACK
#include "flag.h"

#define DATASET_INFO_IMPLEMENTATION
#include "dataset_info.h"

#define BUILD_FOLDER "build/"
#define SRC_FOLDER   "src/"
#define KERNEL_FOLDER "kernels/"

typedef struct {
    char*         target;
    bool          release;
    bool          debug;
    bool          asan;
    bool          omp_off;
    bool          asm_output;
    char*         out_dir;
    Flag_List_Mut macros;

    bool          run;
    bool          slurm;
    Flag_List_Mut partitions;
    char*         script;

    char*         dataset;
    char*         data_dir;

    bool          etags;
    bool          help;
    bool          clean;

    int           rest_argc;
    char**        rest_argv;
} Flags;

Flags flags = {0};
Nob_Cmd cmd = {0};
Nob_Procs procs = {0};

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

size_t ptrlen(void **arr) {
    size_t n = 0;
    while (arr[n]) n++;
    return n;
}

bool mkdir_recursive(const char *path)
{
    if (path == NULL || path[0] == '\0')
    {
        nob_log(NOB_ERROR, "cannot create directory from empty path");
        return false;
    }

    struct stat st;
    bool already_exists = (stat(path, &st) == 0 && S_ISDIR(st.st_mode));
    if (already_exists) return true;

    Nob_String_View sv = nob_sv_from_cstr(path);
    Nob_String_Builder sb = {0};

    while (sv.count > 0)
    {
        Nob_String_View dir = nob_sv_chop_by_delim(&sv, '/');
        if (dir.count == 0) continue;

        nob_sb_appendf(&sb, "%s/", nob_temp_sv_to_cstr(dir));
        int result = mkdir(sb.items, 0755);
        if (result < 0 && errno != EEXIST)
        {
            nob_log(NOB_ERROR, "could not create directory `%s`: %s", sb.items, strerror(errno));
            return false;
        }
    }

    if (!already_exists)
    {
        nob_log(NOB_INFO, "Created directory %s", path);
    }

    return true;
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

size_t gz_decompress(const char* file_path, uint8_t **buf)
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
    uint8_t *input;
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
        for (size_t i = 0; i < total; i++) d[i] = s[i];
    }
    // f4 -> u4
    else if (hdr.type_char == 'f' && hdr.elem_size == 4 && dst_elem_size == 4)
    {
        float *s = (float*)src; uint32_t *d = (uint32_t*)output;
        for (size_t i = 0; i < total; i++) d[i] = s[i];
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

int prepare_dataset()
{
    DatasetInfo *ds_info = &ds_infos[str_to_dataset_kind(flags.dataset)];
    if (!mkdir_recursive(flags.data_dir)) return EXIT_FAILURE;

    const char *zip_name = nob_path_name(ds_info->url);
    const char *zip_path = nob_temp_sprintf("%s/%s", flags.data_dir, zip_name);
    const char *ds_path = path_join(flags.data_dir, ds_info->dir_name);
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
            nob_cmd_append(&cmd, "wget", "-q", "--show-progress", "-P", flags.data_dir, ds_info->url);
            if (!nob_cmd_run(&cmd))
            {
                nob_log(NOB_ERROR, "Failed to download %s dataset", ds_info->name);
                return EXIT_FAILURE;
            }
        }

        nob_log(NOB_INFO, "Extracting %s...", ds_info->name);
        nob_cmd_append(&cmd, "unzip", "-q", "-n", zip_path, "-d", flags.data_dir);
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

void list_datasets(void)
{
    printf("Available datasets:\n");
    for (size_t i = 0; i < NOB_ARRAY_LEN(ds_infos); i++)
    {
        printf("  %s\n", ds_infos[i].name);
    }
}

#define STRINGS(...) { \
    .items = (const char*[]){__VA_ARGS__}, \
    .count = sizeof((const char*[]){__VA_ARGS__}) / sizeof(const char*), \
    .capacity = 0 \
}

// Strings are meant to be stored on stack, and are not
// supposed to be a /Dynamic Array/
typedef struct {
    const char **items;
    size_t count;
    size_t capacity;
} Strings;

typedef struct {
    const char* name;
    Strings srcs;
    Strings libs;
    const char*  out_dir;       // NULL means BUILD_FOLDER
    Strings release_macros;
} Target;

// Target registry
Target targets[] = {
    {
        .name = "paragnn",
        .srcs = STRINGS(
            SRC_FOLDER"main.c",
            SRC_FOLDER"core.c",
            SRC_FOLDER"gnn.c",
            SRC_FOLDER"dataset.c",
            SRC_FOLDER"layers.c",
            SRC_FOLDER"optim.c",
            SRC_FOLDER"linalg/axpy.c",
            SRC_FOLDER"linalg/copy.c",
            SRC_FOLDER"linalg/scal.c",
            SRC_FOLDER"linalg/gemm.c",
            SRC_FOLDER"timer.c",
            ),
        .libs = STRINGS("-lm", "-lopenblas"),
    },
    {
        .name = "sageconv_backward",
        .srcs = STRINGS(
            KERNEL_FOLDER"sageconv_backward.c",
            SRC_FOLDER"core.c",
            SRC_FOLDER"dataset.c",
            SRC_FOLDER"layers.c",
            SRC_FOLDER"timer.c",
            KERNEL_FOLDER"cache_counter.c",
            ),
        .libs = STRINGS("-lm", "-lopenblas"),
    },
    {
        .name = "tsmm_tn",
        .srcs = STRINGS(
            SRC_FOLDER"timer.c",
            KERNEL_FOLDER"cache_counter.c",
            KERNEL_FOLDER"tsmm_tn.c",
            SRC_FOLDER"linalg/gemm.c",
            ),
        .libs = STRINGS("-lm", "-lopenblas"),
    },
    {
        .name = "aggregate",
        .srcs = STRINGS(
            SRC_FOLDER"timer.c",
            SRC_FOLDER"dataset.c",
            KERNEL_FOLDER"cache_counter.c",
            KERNEL_FOLDER"aggregate.c",
            ),
        .libs = STRINGS("-lm"),
    },

};

typedef enum {
    NO_DEFAULT = 0,
    DEFAULT,
} Default;

typedef struct {
    const char *name;
    const char *desc;
    const char *arch;
    Default is_default;
} Partition;

static const Partition partitions[] =  {
    {"defq",     "DP AMD EPYC 7601 32-Core Processor SMT2 128 threads (Zen1)", "x86-64",  NO_DEFAULT},
    {"armq",     "DP Cavium ThunderX2 CN9980 SMT4 256 threads",                "aarch64", NO_DEFAULT},
    {"huaq",     "DP Huawei Kunpeng920-6426 no-HT 128 cores",                  "aarch64", NO_DEFAULT},
    {"milanq",   "DP AMD EPYC 7763 64-Core Processor SMT2 256 threads (Zen3)", "x86-64",  DEFAULT},
    {"fpgaq",    "DP AMD EPYC 7413 24-Core Processor SMT2 96 threads (Zen3)",  "x86-64",  DEFAULT},
    {"genoaxq",  "DP AMD EPYC Genoa-X 9684X 96-Core (SMT2) (Zen4)",            "x86-64",  DEFAULT},
    {"xeonmaxq", "DP Intel XeonMax 9480 56-core (SMT2 144)",                   "x86-64",  DEFAULT},
    {"rome16q",  "SP AMD EPYC 7302P 16-Core Processor SMT2 32 threads (Zen2)", "x86-64",  NO_DEFAULT},
    {"gh200q",   "Nvidia Grace Hopper GH200 APU 72-core cpu",                  "aarch64", DEFAULT},
    {"habanaq",  "DP Intel Xeon Scalable Platinum 8360Y",                      "x86-64",  DEFAULT},
};

void list_partitions(void)
{
    printf("Valid partitions (* = default):\n");
    for (size_t i = 0; i < NOB_ARRAY_LEN(partitions); i++) {
        printf("  %c %-10s %-7s  %s\n",
               partitions[i].is_default ? '*' : ' ',
               partitions[i].name,
               partitions[i].arch,
               partitions[i].desc);

    }
}

bool partition_is_valid(const char* name)
{

    for (size_t i = 0; i < NOB_ARRAY_LEN(partitions); i++) {
        if (strcmp(partitions[i].name, name) == 0) return true;
    }
    return false;
}

Target* find_target(const char* name)
{
    for (size_t i = 0; i < NOB_ARRAY_LEN(targets); i++) {
        if (strcmp(targets[i].name, name) == 0) {
            return &targets[i];
        }
    }
    return NULL;
}

void list_targets(void)
{
    printf("Available targets:\n");
    for (size_t i = 0; i < NOB_ARRAY_LEN(targets); i++) {
        printf("  %s%s\n", targets[i].name,
               i == 0 ? " (default)" : "");
    }
}

typedef enum {
    CPU_UNKNOWN,
    CPU_ARM_NEOVERSE_V2,
    CPU_ARM_THUNDERX2,
    CPU_ARM_KUNPENG_920,
} CPUVendor;

char const* arch_macro_name[] = {
    [CPU_UNKNOWN] = NULL,
    [CPU_ARM_NEOVERSE_V2] = "__neoverse_v2__",
    [CPU_ARM_THUNDERX2] = "__thunderx2__",
    [CPU_ARM_KUNPENG_920] = "__kunpeng_920__",
};

static CPUVendor detect_cpu_vendor(void)
{
#if defined(__aarch64__) || defined(__arm__)
    // ref: https://developer.arm.com/documentation/107771/0102/AArch64-registers/AArch64-Identification-registers-summary/MIDR-EL1--Main-ID-Register
    uint64_t midr;
    __asm__ __volatile__("mrs %0, midr_el1" : "=r"(midr));

    uint8_t implementer = (midr >> 24) & 0xFF;
    uint16_t part_num = (midr >> 4) & 0xFFF;

    // eX3 has only these ARM CPUs, and I don't think they will add any new ones
    // before I finish my thesis. So this should be enough.

    if (implementer == 0x41 && part_num == 0xD4F) return CPU_ARM_NEOVERSE_V2;
    if (implementer == 0x43 && part_num == 0x0AF) return CPU_ARM_THUNDERX2;
    if (implementer == 0x48 && part_num == 0xD01) return CPU_ARM_KUNPENG_920;
#endif
    return CPU_UNKNOWN;
}

const char* get_artifact_path(const char* out_dir, const char* src_path)
{
    const char* base = nob_path_name(src_path);
    Nob_String_View sv = nob_sv_from_cstr(base);

    // Strip .c extension
    size_t len = sv.count;
    if (len > 2 && sv.data[len-2] == '.' && sv.data[len-1] == 'c') {
        len -= 2;
    }

    const char *ext = flags.asm_output ? ".s" : ".o";
    return nob_temp_sprintf("%s%.*s%s", out_dir, (int)len, sv.data, ext);
}

typedef enum {
    COMPILING,
    LINKING,
} BuildPhase;

void append_common_flags(BuildPhase phase)
{
    nob_cc_flags(&cmd);
    nob_cc_error_flags(&cmd);
    nob_cmd_append(&cmd, "-I"SRC_FOLDER);

    if (flags.debug) {
        nob_cmd_append(&cmd, "-ggdb", "-g3", "-gdwarf-2");
    }

    // TODO: Remove this
    nob_cmd_append(&cmd, "-DUSE_DOUBLE");
    nob_cmd_append(&cmd, "-march=native");

    if (flags.release) {
        nob_cmd_append(&cmd, "-O3", "-DNDEBUG", "-ffast-math");
    } else {
        nob_cmd_append(&cmd, "-Og");
    }

    if (flags.asan)
    {
        nob_cmd_append(&cmd, "-fsanitize=address", "-fno-omit-frame-pointer");
    }

    if (phase == COMPILING && flags.omp_off) {
        nob_cmd_append(&cmd, "-Wno-unknown-pragmas");
    } else {
        nob_cmd_append(&cmd, "-fopenmp");
    }
}

int build_target(Target* t)
{
    // Determine output directory
    const char* out_dir = flags.out_dir ? flags.out_dir :
                          t->out_dir    ? t->out_dir : BUILD_FOLDER;

    // Ensure trailing slash
    size_t len = strlen(out_dir);
    if (len > 0 && out_dir[len-1] != '/') {
        out_dir = nob_temp_sprintf("%s/", out_dir);
    }

    if (!mkdir_recursive(out_dir)) return 1;

    const char* exec_path = nob_temp_sprintf("%s%s", out_dir, t->name);

    // Compile all source files
    const char** dst_paths = nob_temp_alloc(t->srcs.count * sizeof(char*));

    CPUVendor cpu_vendor = detect_cpu_vendor();
    if (cpu_vendor != CPU_UNKNOWN)
    {
        nob_log(NOB_INFO, "Adding missing architecture macro %s", arch_macro_name[cpu_vendor]);
    }

    printf("[>>>] Compiling\n");
    for (size_t i = 0; i < t->srcs.count; i++) {
        const char* src = t->srcs.items[i];
        const char* dst = get_artifact_path(out_dir, src);
        dst_paths[i] = dst;

        nob_cc(&cmd);
        append_common_flags(COMPILING);
        if (cpu_vendor != CPU_UNKNOWN)
        {
            nob_cmd_append(&cmd, nob_temp_sprintf("-D%s", arch_macro_name[cpu_vendor]));
        }

        if (flags.release && t->release_macros.items)
        {
            for (size_t i = 0; i < t->release_macros.count; i++)
            {
                bool overwrite = false;
                for (size_t j = 0; j < flags.macros.count; j++)
                {
                    char *user_macro = nob_temp_sprintf("-D%s", flags.macros.items[j]);
                        if (strcmp(user_macro, t->release_macros.items[i]) == 0)
                        {
                            overwrite = true;
                            break;
                        }
                }
                if (!overwrite) nob_cmd_append(&cmd, t->release_macros.items[i]);
            }
        }

        for (size_t j = 0; j < flags.macros.count; j++) {
            nob_cmd_append(&cmd, nob_temp_sprintf("-D%s", flags.macros.items[j]));
        }

        nob_cc_output(&cmd, dst);

        if (flags.asm_output) {
            nob_cmd_append(&cmd, "-S", src);
            nob_cmd_append(&cmd, "-fverbose-asm");
        } else {
            nob_cmd_append(&cmd, "-c", src);
        }

        if (!nob_cmd_run(&cmd, .async = &procs)) {
            nob_log(NOB_ERROR, "Failed to compile %s", src);
            return 1;
        }
    }

    if (!nob_procs_flush(&procs)) return 1;

    if (flags.asm_output) {
        nob_log(NOB_INFO, "Assembly files written to %s", out_dir);
        return 0;
    }

    // Link
    printf("[>>>] Linking\n");
    nob_cc(&cmd);
    append_common_flags(LINKING);
    nob_cmd_append(&cmd, "-fopenmp");

    nob_cc_output(&cmd, exec_path);

    for (size_t i = 0; i < t->srcs.count; i++) {
        nob_cc_inputs(&cmd, dst_paths[i]);
    }

    for (size_t i = 0; i < t->libs.count; i++) {
        nob_cmd_append(&cmd, t->libs.items[i]);
    }

    if (!nob_cmd_run(&cmd)) return 1;

    nob_log(NOB_INFO, "Successfully compiled: %s", exec_path);

    // Run if requested
    if (flags.run) {
        nob_cmd_append(&cmd, exec_path);
        for (int i = 0; i < flags.rest_argc; i++) {
            nob_cmd_append(&cmd, flags.rest_argv[i]);
        }
        if (flags.dataset) nob_cmd_append(&cmd, "-dataset", flags.dataset);
        if (flags.data_dir) nob_cmd_append(&cmd, "-data-dir", flags.data_dir);
        if (!nob_cmd_run(&cmd)) return 1;
    }

    return 0;
}

const char** resolve_partitions(size_t *count)
{
    static const char* result[NOB_ARRAY_LEN(partitions)];
    *count = 0;

    // No partitions specified: default to bench
    if (flags.partitions.count == 0) {
        for (size_t i = 0; i < NOB_ARRAY_LEN(partitions); i++) {
            if (partitions[i].is_default) result[(*count)++] = partitions[i].name;
        }
        return result;
    }

    if (strcmp(flags.partitions.items[0], "all") == 0) {
        for (size_t i = 0; i < NOB_ARRAY_LEN(partitions); i++) {
            result[(*count)++] = partitions[i].name;
        }
        return result;
    }

    if (strcmp(flags.partitions.items[0], "x86-64") == 0) {
        for (size_t i = 0; i < NOB_ARRAY_LEN(partitions); i++) {
            if (strcmp(partitions[i].arch, "x86-64") == 0) result[(*count)++] = partitions[i].name;
        }
        return result;
    }

    if (strcmp(flags.partitions.items[0], "aarch64") == 0) {
        for (size_t i = 0; i < NOB_ARRAY_LEN(partitions); i++) {
            if (strcmp(partitions[i].arch, "aarch64") == 0) result[(*count)++] = partitions[i].name;
        }
        return result;
    }

    bool found_invalid = false;
    // Check if partitions where given as comma separated list
    if (strchr(flags.partitions.items[0], ',')) {
        char *p = flags.partitions.items[0], *tok;
        while ((tok = strsep(&p, ","))) {
            if (*tok) {
                if (!partition_is_valid(tok)) {
                    nob_log(NOB_ERROR, "Unknown partition: %s", tok);
                    found_invalid = true;
                } else {
                    result[(*count)++] = tok;
                }
            }
        }
        goto leave;
    }

    for (size_t j = 0; j < flags.partitions.count; j++) {
        const char *p = flags.partitions.items[j];
        if (!partition_is_valid(p)) {
            nob_log(NOB_ERROR, "Unknown partition: %s", p);
            found_invalid = true;
        } else {
            result[(*count)++] = p;
        }
    }

leave:
    if (found_invalid) {
        list_partitions();
        return NULL;
    }

    return result;
}

const char* get_config_suffix(void)
{
    if (flags.macros.count == 0) return "default";

    Nob_String_Builder sb = {0};
    for (size_t i = 0; i < flags.macros.count; i++) {
        if (i > 0) nob_sb_append_cstr(&sb, "-");
        // Strip "USE_" prefix if present for brevity
        const char *m = flags.macros.items[i];
        if (strncmp(m, "USE_", 4) == 0) m += 4;
        // Convert to lowercase
        for (; *m; m++) nob_da_append(&sb, tolower(*m));
    }
    nob_sb_append_null(&sb);
    return nob_temp_strdup(sb.items);
}

const char *parse_sbatch_job_name(const char *script)
{
    Nob_String_Builder sb = {0};
    if (!nob_read_entire_file(script, &sb)) return NULL;
    nob_sb_append_null(&sb);

    const char *needle = "#SBATCH --job-name=";
    char *p = strstr(sb.items, needle);
    if (!p) return NULL;
    p += strlen(needle);

    char quote = 0;
    if (*p == '"' || *p == '\'') { quote = *p; p++; }

    const char *start = p;
    while (*p && *p != '\n' && *p != ' ' && *p != quote) p++;

    return nob_temp_sprintf("%.*s", (int)(p - start), start);
}

int submit_slurm(void)
{
    const char *config = get_config_suffix();
    time_t now = time(NULL);
    struct tm *local_time = localtime(&now);
    char date_string[20];
    strftime(date_string, sizeof(date_string), "%Y%m%d-%H%M%S", local_time);
    const char *script = flags.script ? flags.script : "run_benchmark.sbatch";
    const char *job_name = flags.script ? parse_sbatch_job_name(script) : NULL;
    const char *label = job_name ? job_name : nob_temp_sprintf("%s-%s", flags.target, config);
    const char *log_dir = nob_temp_sprintf("logs/%s/%s", date_string, label);
    if (!mkdir_recursive(log_dir)) return 1;

    size_t count;
    const char **parts = resolve_partitions(&count);
    if (!parts) return 1;


    Nob_String_Builder macros_str = {0};
    for (size_t i = 0; i < flags.macros.count; i++) {
        if (i > 0) nob_sb_append_cstr(&macros_str, " ");
        nob_sb_append_cstr(&macros_str, nob_temp_sprintf("-D %s", flags.macros.items[i]));
    }
    if (macros_str.count == 0) nob_sb_append_null(&macros_str);

    int ret = 0;
    for (size_t i = 0; i < count; i++) {
        nob_cmd_append(&cmd, "sbatch", "-p", parts[i]);
        if (!flags.script) {
            nob_cmd_append(&cmd, nob_temp_sprintf("--export=BENCHMARK_TARGET=%s,BENCHMARK_CONFIG=%s,BENCHMARK_MACROS=%s",
                                                  flags.target, config, macros_str.items));
            nob_cmd_append(&cmd, "-J", nob_temp_sprintf("%s-%s", flags.target, config));
        }
        nob_cmd_append(&cmd, "-o", nob_temp_sprintf("%s/%s.out", log_dir, parts[i]));
        nob_cmd_append(&cmd, script);
        nob_cmd_append(&cmd, "--exclusive");
        nob_cmd_append(&cmd, "run_benchmark.sbatch");

        if (!nob_cmd_run(&cmd)) {
            nob_log(NOB_ERROR, "Failed to submit %s to %s", script, parts[i]);
            ret = 1;
        } else {
            nob_log(NOB_INFO, "Submitted %s to %s", script, parts[i]);
        }
    }

    return ret;
}

int clean(void)
{
    nob_log(NOB_INFO, "Cleaning build artifacts...");
    nob_delete_file(BUILD_FOLDER);
    return 0;
}

int etags(void)
{
    const char *folders[] = { SRC_FOLDER, KERNEL_FOLDER };

    nob_log(NOB_INFO, "Generating etags...");

    Nob_Cmd cmd = {0};
    nob_cmd_append(&cmd, "find", ".", "-type", "f");

    nob_cmd_append(&cmd, "(");
    for (size_t i = 0; i < NOB_ARRAY_LEN(folders); i++) {
        if (i > 0) nob_cmd_append(&cmd, "-o");
        nob_cmd_append(&cmd, "-ipath",
                       nob_temp_sprintf("*/%s*", folders[i]));
    }
    nob_cmd_append(&cmd, ")");

    nob_cmd_append(&cmd, "(", "-name", "*.[ch]", ")");
    nob_cmd_append(&cmd, "-exec", "etags", "--declarations", "{}", "+");
    if (!nob_cmd_run(&cmd)) return 1;
    return 0;
}

void usage(FILE* stream)
{
    fprintf(stream, "Usage: ./nob [OPTIONS] [--] [ARGS]\n\n");
    fprintf(stream, "OPTIONS:\n");
    flag_print_options(stream);
    fprintf(stream, "\nDATASETS:\n");
    list_datasets();
    fprintf(stream, "\nTARGETS:\n");
    list_targets();
}

int main(int argc, char** argv)
{
#ifndef NOB_NO_REBUILD
    NOB_GO_REBUILD_URSELF(argc, argv);
#endif

    flag_str_var(&flags.target,          "target",       "paragnn", "Build target (see list below)");
    flag_bool_var(&flags.release,        "release",      false,     "Build in release mode");
    flag_bool_var(&flags.debug,          "debug",        false,     "Build in debug mode (default)");
    flag_bool_var(&flags.asan,           "asan",         false,     "Enable AddressSanitizer");
    flag_bool_var(&flags.omp_off,        "omp-off",      false,     "Don't compile with -fopenmp");
    flag_bool_var(&flags.asm_output,     "S",            false,     "Produce assembly instead of object files");
    flag_str_var(&flags.out_dir,         "o",            NULL,      "Output directory");
    flag_list_mut_var(&flags.macros,     "D",                       "Define macro (e.g., -D SIMD_ENABLED)");

    flag_bool_var(&flags.run,            "run",          false,     "Run after building");
    flag_bool_var(&flags.slurm,          "slurm",        false,     "Submit job to Slurm");
    flag_list_mut_var(&flags.partitions, "p",                       "Partition(s) to submit to (or 'list', 'all', 'aarch64', 'x86_64')");
    flag_str_var(&flags.script, "script", NULL, "Sbatch script to submit (e.g., adhoc.sh)");

    flag_str_var(&flags.dataset,         "dataset",      "arxiv",   "Dataset to use (arxiv, products, papers100M)");
    flag_str_var(&flags.data_dir,        "data-dir",     "./data",  "Directory for downloading and reading datasets");

    flag_bool_var(&flags.clean,          "clean",        false,     "Clean build artifacts");
    flag_bool_var(&flags.etags,          "etags",        false,     "Genereate etags");
    flag_bool_var(&flags.help,           "help",         false,     "Print this help message");

    if (!flag_parse(argc, argv)) {
        usage(stderr);
        flag_print_error(stderr);
        return 1;
    }

    if (flags.help) {
        usage(stdout);
        return 0;
    }

    if (strcmp(flags.dataset, "list") == 0)
    {
        list_datasets();
        return 0;
    }
    if (str_to_dataset_kind(flags.dataset) == DATASET_INVALID)
    {
        nob_log(NOB_ERROR, "Given dataset is not valid: %s", flags.dataset);
        list_datasets();
        return 1;
    }

    // Handle rest args
    argc = flag_rest_argc();
    argv = flag_rest_argv();
    if (argc > 0 && strcmp(argv[0], "--") == 0) {
        argv++;
        argc--;
    }
    flags.rest_argc = argc;
    flags.rest_argv = argv;

    // Default to debug if neither specified
    if (!flags.release && !flags.debug) {
        flags.debug = true;
    }

    // Commands
    if (flags.clean) {
        return clean();
    }

    if (flags.etags) {
        return etags();
    }

    if (flags.partitions.count > 0 && strcmp(flags.partitions.items[0], "list") == 0) {
        list_partitions();
        return 0;
    }

    if (flags.slurm) {
        return submit_slurm();
    }

    // Build target
    Target* t = find_target(flags.target);
    if (t == NULL) {
        nob_log(NOB_ERROR, "Unknown target: %s", flags.target);
        list_targets();
        return 1;
    }

    if (prepare_dataset() == EXIT_FAILURE) return EXIT_FAILURE;
    return build_target(t);
}
