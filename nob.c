#include <getopt.h>
#include <unistd.h>

#define NOB_EXPERIMENTAL_DELETE_OLD
#define NOB_IMPLEMENTATION
#define NOB_WARN_DEPRECATED
#define NOB_REBUILD_URSELF(binary_path, source_path) \
    "gcc", "-ggdb", "-o", binary_path, source_path
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

#define BUILD_FOLDER "build/"
#define SRC_FOLDER   "src/"
#define KERNEL_FOLDER "kernels/"

enum {
    OK        = 0,
    ERR       = 1,
    ERR_USAGE = 2,
};

typedef struct {
    char **items;
    size_t count;
    size_t capacity;
} Strings;

typedef enum {
    NAIVE_IMPL,
    BLAS_IMPL,
    TUNED_IMPL,
} Impl;

// Default flag values
#define DEFAULT_DATASET "arxiv"
#define DEFAULT_TARGET  "paragnn"
#define DEFAULT_DATADIR "~/D1/paragnn-dataset"
#define DEFAULT_IMPL    BLAS_IMPL // TODO: change to tuned

typedef struct {
    char*   target;
    bool    release;
    bool    debug;
    bool    asan;
    bool    omp_off;
    bool    use_dbl;
    bool    asm_output;
    char*   out_dir;
    Strings macros;
    Impl impl;

    bool    run;
    bool    slurm;
    Strings partitions;
    char*   script;

    char*   dataset;
    char*   datadir;

    bool    etags;
    bool    help;
    bool    clean;

    int     rest_argc;
    char**  rest_argv;
} Flags;

enum {
    // w/ shorthands
    OPT_TARGET  = 't',
    OPT_RELEASE = 'r',
    OPT_DEBUG   = 'g',
    OPT_OUTDIR  = 'o',
    OPT_ASAN    = 'a',
    OPT_ASM     = 'S',
    OPT_IMPL    = 'i',
    OPT_HELP    = 'h',
    // w/o shorthands
    OPT_OMP_OFF = 256,          // above ASCII
    OPT_DBL,
    OPT_MACRO,
    OPT_RUN,
    OPT_SLURM,
    OPT_PARTITION,
    OPT_SCRIPT,
    OPT_DATASET,
    OPT_DATADIR,
    OPT_ETAGS,
    OPT_CLEAN,
};

static struct option long_options[] = {
    {"target",  required_argument, NULL, OPT_TARGET},
    {"release", no_argument,       NULL, OPT_RELEASE},
    {"debug",   no_argument,       NULL, OPT_DEBUG},
    {"asan",    no_argument,       NULL, OPT_ASAN},
    {"omp-off", no_argument,       NULL, OPT_OMP_OFF},
    {"S",       no_argument,       NULL, OPT_ASM},
    {"double",  no_argument,       NULL, OPT_DBL},
    {"o",       required_argument, NULL, OPT_OUTDIR},
    {"D",       required_argument, NULL, OPT_MACRO},
    {"impl",    required_argument, NULL, OPT_IMPL},
    {"run",     no_argument,       NULL, OPT_RUN},
    {"slurm",   no_argument,       NULL, OPT_SLURM},
    {"p",       required_argument, NULL, OPT_PARTITION},
    {"script",  required_argument, NULL, OPT_SCRIPT},
    {"dataset", required_argument, NULL, OPT_DATASET},
    {"datadir", required_argument, NULL, OPT_DATADIR},
    {"etags",   no_argument,       NULL, OPT_ETAGS},
    {"help",    no_argument,       NULL, OPT_HELP},
    {"clean",   no_argument,       NULL, OPT_CLEAN},
    {0,         0,                 0,    0}
};

Flags     flags = {0};
Nob_Cmd   cmd   = {0};
Nob_Procs procs = {0};

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

// Strings are meant to be stored on stack, and are not
// supposed to be a /Dynamic Array/
#define STRS_STATIC(...) { \
    .items = (char*[]){__VA_ARGS__}, \
    .count = sizeof((char*[]){__VA_ARGS__}) / sizeof(char*), \
    .capacity = 0 \
}

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
        .srcs = STRS_STATIC(
            SRC_FOLDER"main.c",
            SRC_FOLDER"core.c",
            SRC_FOLDER"nn.c",
            SRC_FOLDER"sageconv.c",
            SRC_FOLDER"matmul_naive.c",
            SRC_FOLDER"dataset.c",
            SRC_FOLDER"dataset_info.c",
            SRC_FOLDER"layers.c",
            SRC_FOLDER"optim.c",
            SRC_FOLDER"timer.c",
            ),
        .libs = STRS_STATIC("-lm", "-lopenblas"),
    },
    {
        .name = "prepare_dataset",
        .srcs = STRS_STATIC(
            SRC_FOLDER"prepare_dataset.c",
            SRC_FOLDER"dataset_info.c",
            ),
        .libs = STRS_STATIC("-lz"),
    },
    {
        .name = "sageconv_backward",
        .srcs = STRS_STATIC(
            KERNEL_FOLDER"sageconv_backward.c",
            KERNEL_FOLDER"sageconv_backward_common.c",
            KERNEL_FOLDER"sageconv_backward_gemm_tn.c",
            KERNEL_FOLDER"sageconv_backward_fused.c",
            SRC_FOLDER"core.c",
            SRC_FOLDER"dataset.c",
            SRC_FOLDER"layers.c",
            SRC_FOLDER"timer.c",
            KERNEL_FOLDER"cache_counter.c",
            ),
        .libs = STRS_STATIC("-lm", "-lopenblas"),
    },
    {
        .name = "aggregate",
        .srcs = STRS_STATIC(
            SRC_FOLDER"timer.c",
            SRC_FOLDER"dataset.c",
            KERNEL_FOLDER"cache_counter.c",
            KERNEL_FOLDER"aggregate.c",
            ),
        .libs = STRS_STATIC("-lm"),
    },
};

Target* find_target(const char* name)
{
    for (size_t i = 0; i < NOB_ARRAY_LEN(targets); i++)
    {
        if (strcmp(targets[i].name, name) == 0)
        {
            return &targets[i];
        }
    }
    return NULL;
}

void list_targets(void)
{
    printf("Available targets:\n");
    for (size_t i = 0; i < NOB_ARRAY_LEN(targets); i++)
    {
        printf("  %s%s\n", targets[i].name, i == 0 ? " (default)" : "");
    }
}

const char* get_artifact_path(const char* out_dir, const char* src_path)
{
    const char *base = nob_path_name(src_path);
    const char *dot = strrchr(base, '.');
    int name_len = dot ? (int)(dot - base) : (int)strlen(base);
    const char *ext = flags.asm_output ? ".s" : ".o";
    return nob_path_join_temp(out_dir, nob_temp_sprintf("%.*s%s", name_len, base, ext));
}

typedef enum {
    COMPILING,
    LINKING,
} BuildPhase;

void append_common_flags(Target* t, BuildPhase phase)
{
    nob_cc_flags(&cmd);
    nob_cc_error_flags(&cmd);
    nob_cmd_append(&cmd, "-I"SRC_FOLDER);

    if (flags.debug)
    {
        nob_cmd_append(&cmd, "-ggdb", "-g3", "-gdwarf-2");
    }

    if (strcmp("paragnn", t->name) == 0)
    {
        if (flags.impl == NAIVE_IMPL) nob_cmd_append(&cmd, "-DSAGECONV_NAIVE_IMPL");
        if (flags.impl == BLAS_IMPL)  nob_cmd_append(&cmd, "-DSAGECONV_BLAS_IMPL");
        if (flags.impl == TUNED_IMPL) nob_cmd_append(&cmd, "-DSAGECONV_TUNED_IMPL");
    }

    if (flags.use_dbl)
    {
        nob_cmd_append(&cmd, "-DUSE_DOUBLE");
    }

    if (flags.release)
    {
        nob_cmd_append(&cmd, "-O3", "-DNDEBUG", "-ffast-math");
        // XXX: building with -march=native on Kunpeng-920 expands to wrong extensions on eX3 for gcc15
        if (detect_cpu_vendor() == CPU_ARM_KUNPENG_920)
            nob_cmd_append(&cmd, "-march=armv8.2-a+dotprod+crc+crypto+fp16fml");
        else
            nob_cmd_append(&cmd, "-march=native");
    }
    else
    {
        nob_cmd_append(&cmd, "-Og");
    }

    if (flags.asan)
    {
        nob_cmd_append(&cmd, "-fsanitize=address", "-fno-omit-frame-pointer");
    }

    if (phase == COMPILING && flags.omp_off)
    {
        nob_cmd_append(&cmd, "-Wno-unknown-pragmas");
    }
    else
    {
        nob_cmd_append(&cmd, "-fopenmp");
    }
}

int build(const char *target_str, bool run)
{
    Target* t = find_target(target_str);
    if (t == NULL)
    {
        nob_log(NOB_ERROR, "unknown target: %s", target_str);
        return ERR_USAGE;
    }

    // Determine output directory
    const char* out_dir = flags.out_dir ? flags.out_dir :
                          t->out_dir    ? t->out_dir : BUILD_FOLDER;

    // Ensure trailing slash
    if (!nob_mkdir_recursive(out_dir)) return ERR;
    const char* exec_path = nob_path_join_temp(out_dir, t->name);

    // Compile all source files
    const char** dst_paths = nob_temp_alloc(t->srcs.count * sizeof(char*));

    CPUVendor cpu_vendor = detect_cpu_vendor();
    if (cpu_vendor != CPU_UNKNOWN)
    {
        nob_log(NOB_INFO, "Adding missing architecture macro %s", arch_macro_name[cpu_vendor]);
    }

    printf("[>>>] Compiling\n");
    for (size_t i = 0; i < t->srcs.count; i++)
    {
        const char* src = t->srcs.items[i];
        const char* dst = get_artifact_path(out_dir, src);
        dst_paths[i] = dst;

        nob_cc(&cmd);
        append_common_flags(t, COMPILING);
        if (cpu_vendor != CPU_UNKNOWN)
        {
            nob_cmd_append(&cmd, nob_temp_sprintf("-D%s", arch_macro_name[cpu_vendor]));
        }

        if (flags.release && t->release_macros.items)
        {            for (size_t i = 0; i < t->release_macros.count; i++)
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

        for (size_t j = 0; j < flags.macros.count; j++)
        {
            nob_cmd_append(&cmd, nob_temp_sprintf("-D%s", flags.macros.items[j]));
        }

        nob_cc_output(&cmd, dst);

        if (flags.asm_output)
        {
            nob_cmd_append(&cmd, "-S", src);
            nob_cmd_append(&cmd, "-fverbose-asm");
        }
        else
        {
            nob_cmd_append(&cmd, "-c", src);
        }

        if (!nob_cmd_run(&cmd, .async = &procs))
        {
            fprintf(stderr, "Error: failed to compile %s", src);
            return ERR;
        }
    }

    if (!nob_procs_flush(&procs)) return ERR;

    if (flags.asm_output)
    {
        nob_log(NOB_INFO, "Assembly files written to %s", out_dir);
        return OK;
    }

    // Link
    printf("[>>>] Linking\n");
    nob_cc(&cmd);
    append_common_flags(t, LINKING);
    nob_cmd_append(&cmd, "-fopenmp");

    nob_cc_output(&cmd, exec_path);

    for (size_t i = 0; i < t->srcs.count; i++)
    {
        nob_cc_inputs(&cmd, dst_paths[i]);
    }

    for (size_t i = 0; i < t->libs.count; i++)
    {
        nob_cmd_append(&cmd, t->libs.items[i]);
    }

    if (!nob_cmd_run(&cmd)) return ERR;

    nob_log(NOB_INFO, "Successfully compiled: %s", exec_path);

    // Run if requested
    if (run)
    {
        nob_cmd_append(&cmd, exec_path);
        for (int i = 0; i < flags.rest_argc; i++)
        {
            nob_cmd_append(&cmd, flags.rest_argv[i]);
        }
        if (flags.dataset) nob_cmd_append(&cmd, "-dataset", flags.dataset);
        if (flags.datadir) nob_cmd_append(&cmd, "-datadir", flags.datadir);
        if (!nob_cmd_run(&cmd)) return ERR;
    }

    return OK;
}

int prepare_dataset()
{
    int rc = ERR;

    rc = build("prepare_dataset", false);
    if (rc != OK)
    {
        nob_log(NOB_ERROR, "Could not build 'prepare_dataset'");
        goto exit;
    }

    const char *bin_dir = flags.out_dir ? flags.out_dir : BUILD_FOLDER;
    const char* bin_path = nob_path_join_temp(bin_dir, "prepare_dataset");
    nob_cmd_append(&cmd, bin_path, "-dataset", flags.dataset, "-datadir", flags.datadir);
    if (!nob_cmd_run(&cmd)) goto exit;
    rc = OK;

exit:
     nob_temp_reset();
    return rc;
}

typedef enum {
    L3_PMU_NO = 0,
    L3_PMU_YES,
} L3MissPMU;

typedef enum {
    GPU_NO = 0,
    GPU_YES,
} GPUNode;

typedef struct {
    const char *name;
    const char *desc;
    const char *arch;
    L3MissPMU   l3_miss_pmu;
    GPUNode     gpu_node;
} Partition;

static const Partition partitions[] =  {
    {"defq",     "DP AMD EPYC 7601 32-Core Processor SMT2 128 threads (Zen1)", "x86-64",  L3_PMU_NO,  GPU_NO},
    {"armq",     "DP Cavium ThunderX2 CN9980 SMT4 256 threads",                "aarch64", L3_PMU_NO,  GPU_NO},
    {"huaq",     "DP Huawei Kunpeng920-6426 no-HT 128 cores",                  "aarch64", L3_PMU_NO,  GPU_NO},
    {"milanq",   "DP AMD EPYC 7763 64-Core Processor SMT2 256 threads (Zen3)", "x86-64",  L3_PMU_YES, GPU_NO},
    {"fpgaq",    "DP AMD EPYC 7413 24-Core Processor SMT2 96 threads (Zen3)",  "x86-64",  L3_PMU_YES, GPU_NO},
    {"genoaxq",  "DP AMD EPYC Genoa-X 9684X 96-Core (SMT2) (Zen4)",            "x86-64",  L3_PMU_YES, GPU_NO},
    {"xeonmaxq", "DP Intel XeonMax 9480 56-core (SMT2 144)",                   "x86-64",  L3_PMU_YES, GPU_NO},
    {"rome16q",  "SP AMD EPYC 7302P 16-Core Processor SMT2 32 threads (Zen2)", "x86-64",  L3_PMU_NO,  GPU_NO},
    {"gh200q",   "Nvidia Grace Hopper GH200 APU 72-core cpu",                  "aarch64", L3_PMU_YES, GPU_NO},
    {"habanaq",  "DP Intel Xeon Scalable Platinum 8360Y",                      "x86-64",  L3_PMU_YES, GPU_NO},
    {"dgx2q",    "Tesla V100-SXM3-32GB",                                       "x68-64",  L3_PMU_NO,  GPU_YES},
    {"a40q",     "NVIDIA A40",                                                 "aarch64", L3_PMU_NO,  GPU_YES},
    {"a100q",    "NVIDIA A100-PCIE-40GB",                                      "x86-64",  L3_PMU_NO,  GPU_YES},
    {"hgx2q",    "NVIDIA A100-SXM4-80GB",                                      "x86-64",  L3_PMU_NO,  GPU_YES},
    {"gh200q",   "",                                                           "aarch64", L3_PMU_NO,  GPU_YES},
};

void list_partitions(void)
{
    printf("Valid partitions (* = has L3 miss PMU):\n");
    for (size_t i = 0; i < NOB_ARRAY_LEN(partitions); i++)
    {
        printf("  %c %-10s %-7s  %s\n",
               partitions[i].l3_miss_pmu ? '*' : ' ',
               partitions[i].name,
               partitions[i].arch,
               partitions[i].desc);

    }
}

bool partition_is_valid(const char* name)
{

    for (size_t i = 0; i < NOB_ARRAY_LEN(partitions); i++)
    {
        if (strcmp(partitions[i].name, name) == 0) return true;
    }
    return false;
}

size_t resolve_partitions(const char** parts)
{
    size_t count = 0;

    // No partitions specified: default to bench
    if (flags.partitions.count == 0)
    {
        for (size_t i = 0; i < NOB_ARRAY_LEN(partitions); i++)
        {
            if (partitions[i].l3_miss_pmu) parts[count++] = partitions[i].name;
        }
        goto exit;
    }

    if (strcmp(flags.partitions.items[0], "cpu") == 0)
    {
        for (size_t i = 0; i < NOB_ARRAY_LEN(partitions); i++)
        {
            if (partitions[i].gpu_node == GPU_NO) parts[count++] = partitions[i].name;
        }
        goto exit;
    }

    if (strcmp(flags.partitions.items[0], "pmu") == 0)
    {
        for (size_t i = 0; i < NOB_ARRAY_LEN(partitions); i++)
        {
            if (partitions[i].l3_miss_pmu == L3_PMU_YES) parts[count++] = partitions[i].name;
        }
        goto exit;
    }

    if (strcmp(flags.partitions.items[0], "gpu") == 0)
    {
        for (size_t i = 0; i < NOB_ARRAY_LEN(partitions); i++)
        {
            if (partitions[i].gpu_node == GPU_YES) parts[count++] = partitions[i].name;
        }
        goto exit;
    }

    if (strcmp(flags.partitions.items[0], "x86-64") == 0)
    {
        for (size_t i = 0; i < NOB_ARRAY_LEN(partitions); i++)
        {
            if (strcmp(partitions[i].arch, "x86-64") == 0) parts[count++] = partitions[i].name;
        }
        goto exit;
    }

    if (strcmp(flags.partitions.items[0], "aarch64") == 0)
    {
        for (size_t i = 0; i < NOB_ARRAY_LEN(partitions); i++)
        {
            if (strcmp(partitions[i].arch, "aarch64") == 0) parts[count++] = partitions[i].name;
        }
        goto exit;
    }

    // Check if partitions where given as comma separated list
    if (strchr(flags.partitions.items[0], ','))
    {
        char *p = flags.partitions.items[0], *tok;
        while ((tok = strsep(&p, ",")))
        {
            if (*tok)
            {
                if (!partition_is_valid(tok))
                {
                    nob_log(NOB_ERROR, "Unknown partition: %s", tok);
                    count = 0;
                    goto exit;
                }
                else
                {
                    parts[count++] = tok;
                }
            }
        }
        goto exit;
    }

    for (size_t j = 0; j < flags.partitions.count; j++)
    {
        const char *p = flags.partitions.items[j];
        if (!partition_is_valid(p))
        {
            nob_log(NOB_ERROR, "Unknown partition: %s", p);
            count = 0;
            goto exit;
        }
        else
        {
            parts[count++] = p;
        }
        goto exit;
    }

exit:
    return count;
}

const char* get_config_suffix(void)
{
    if (flags.macros.count == 0) return "default";

    Nob_String_Builder sb = {0};
    for (size_t i = 0; i < flags.macros.count; i++)
    {
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

const char *capture_cmd(const char *command)
{
    FILE *fp = popen(command, "r");
    if (!fp) return NULL;
    char buf[128] = {0};
    if (!fgets(buf, sizeof(buf), fp)) { pclose(fp); return NULL; }
    pclose(fp);
    char *nl = strchr(buf, '\n');
    if (nl) *nl = '\0';
    return nob_temp_strdup(buf);
}

const char *make_timestamp(void)
{
    time_t now = time(NULL);
    struct tm *t = localtime(&now);
    char *ts = nob_temp_alloc(20);
    strftime(ts, 20, "%Y%m%dT%H%M%S", t);
    return ts;
}

// Build a label like "paragnn" or "paragnn-tsmm" depending on config
const char *make_label(void)
{
    const char *config = get_config_suffix();
    if (strcmp(config, "default") == 0) return flags.target;
    return nob_temp_sprintf("%s-%s", flags.target, config);
}

int submit_slurm(const char *partition)
{
    int rc = ERR;

    const char *hash = capture_cmd("git rev-parse --short HEAD");
    const char *ts_hash = nob_temp_sprintf("%s-%s", make_timestamp(), hash ? hash : "unknown");

    const char *script = flags.script ? flags.script : "run_benchmark.sbatch";
    const char *label = flags.script ? parse_sbatch_job_name(script) : make_label();

    const char *state = getenv("XDG_STATE_HOME");
    const char *base = state ? state : nob_path_join_temp(getenv("HOME"), ".local/state");
    const char *log_dir = nob_path_join_tempv(base, "paragnn/sbatch-logs", label, ts_hash, NULL);
    if (!nob_mkdir_recursive(log_dir)) goto exit;

    const char *config = get_config_suffix();
    Nob_String_Builder macros_str = {0};
    for (size_t i = 0; i < flags.macros.count; i++)
    {
        if (i > 0) nob_sb_append_cstr(&macros_str, " ");
        nob_sb_append_cstr(&macros_str, nob_temp_sprintf("-D %s", flags.macros.items[i]));
    }
    if (macros_str.count == 0) nob_sb_append_null(&macros_str);

    int ret = 0;
    nob_cmd_append(&cmd, "sbatch", "-p", partition);
    if (!flags.script)
    {
        nob_cmd_append(&cmd, nob_temp_sprintf("--export=BENCHMARK_TARGET=%s,BENCHMARK_CONFIG=%s,BENCHMARK_MACROS=%s",
                                              flags.target, config, macros_str.items));
        nob_cmd_append(&cmd, "-J", nob_temp_sprintf("%s-%s", flags.target, config));
    }
    nob_cmd_append(&cmd, "-o", nob_temp_sprintf("%s/%s.out", log_dir, partition));
    nob_cmd_append(&cmd, "-e", nob_temp_sprintf("%s/%s.err", log_dir, partition));
    nob_cmd_append(&cmd, "--exclusive");
    nob_cmd_append(&cmd, script);

    if (!nob_cmd_run(&cmd))
    {
        nob_log(NOB_ERROR, "Failed to submit %s to %s", script, partition);
        goto exit;
    }
    else
    {
        nob_log(NOB_INFO, "Submitted %s to %s", script, partition);
    }

    rc = OK;
exit:
    return rc;
}

int slurm_via_worktree(void)
{
    int rc = ERR;
    const char *hash = capture_cmd("git rev-parse --short HEAD");
    if (!hash) hash = "unknown";

    const char *ts = make_timestamp();
    const char *label = make_label();
    const char *ts_hash = nob_temp_sprintf("%s-%s", ts, hash);
    const char *wt_path_common = nob_path_join_tempv("../wt/", label, ts_hash, NULL);

    const char *selected_partitions[NOB_ARRAY_LEN(partitions)] = {0};
    size_t count = resolve_partitions(selected_partitions);
    if (count == 0)
    {
        rc = ERR_USAGE;
        goto exit;
    }

    for (size_t i = 0; i < count; i++)
    {
        const char *wt_path = nob_path_join_temp(wt_path_common, selected_partitions[i]);
        nob_cmd_append(&cmd, "git", "worktree", "add", "--detach", wt_path, "HEAD");
        if (!nob_cmd_run(&cmd)) goto exit;

        const char *original_dir = nob_get_current_dir_temp();
        if (!nob_set_current_dir(wt_path)) goto exit;

        size_t checkpoint = nob_temp_save();
        rc = submit_slurm(selected_partitions[i]);
        nob_temp_rewind(checkpoint);

        nob_set_current_dir(original_dir);
        if (rc != OK) goto exit;
        nob_log(NOB_INFO, "Worktree: %s", wt_path);
    }

    rc = OK;

exit:
    nob_temp_reset();
    return rc;
}

int clean(void)
{
    nob_log(NOB_INFO, "Cleaning build artifacts...");
    bool success = nob_delete_file(BUILD_FOLDER);
    if (!success) return ERR;
    return OK;
}

int etags(void)
{
    int rc = ERR;

    const char *folders[] = { SRC_FOLDER, KERNEL_FOLDER };

    nob_log(NOB_INFO, "Generating etags...");

    Nob_Cmd cmd = {0};
    nob_cmd_append(&cmd, "find", ".", "-type", "f");

    nob_cmd_append(&cmd, "(");
    for (size_t i = 0; i < NOB_ARRAY_LEN(folders); i++)
    {
        if (i > 0) nob_cmd_append(&cmd, "-o");
        nob_cmd_append(&cmd, "-ipath", nob_temp_sprintf("*/%s*", folders[i]));
    }
    nob_cmd_append(&cmd, ")");

    nob_cmd_append(&cmd, "(", "-name", "*.[ch]", ")");
    nob_cmd_append(&cmd, "-exec", "etags", "--declarations", "{}", "+");
    if (!nob_cmd_run(&cmd)) goto exit;
    rc = OK;

exit:
    nob_temp_reset();
    return rc;
}

void usage(const char *progname)
{
    fprintf(stderr,
            "Usage: %s [OPTIONS] [--] [ARGS]\n"
            "\n"
            "Build options:\n"
            "  -t, -target NAME   Build target (default: %s)\n"
            "  -r, -release       Build in release mode\n"
            "  -g, -debug         Build in debug mode (default if neither specified)\n"
            "  -a, -asan          Enable AddressSanitizer\n"
            "  -S                 Produce assembly instead of object files\n"
            "  -o DIR             Output directory\n"
            "  -double            Use double precision\n"
            "  -D MACRO[=VAL]     Define macro, repeatable\n"
            "  -omp-off           Don't compile with -fopenmp\n"
            "\n"
            "Run options:\n"
            "  -run               Run after building\n"
            "  -slurm             Submit job to Slurm (creates a git worktree snapshot; commit first)\n"
            "  -p PARTITION       Partition(s), repeatable (or 'list', 'cpu', 'gpu', 'pmu', 'aarch64', 'x86-64')\n"
            "  -script FILE       Sbatch script to submit (e.g., adhoc.sh)\n"
            "\n"
            "Data options:\n"
            "  -dataset NAME      Dataset {arxiv, products, papers100M} (default: %s)\n"
            "  -datadir PATH      Data directory (default: %s)\n"
            "\n"
            "Other:\n"
            "  -etags             Generate etags\n"
            "  -clean             Clean build artifacts\n"
            "  -h, -help          Print this help message\n"
            "\n"
            "Targets:\n",
            progname, flags.target, flags.dataset, flags.datadir);
    list_targets();
    fprintf(stderr, "\nPartitions:\n");
    list_partitions();
}

int main(int argc, char** argv)
{
#ifndef NOB_NO_REBUILD
    NOB_GO_REBUILD_URSELF(argc, argv);
#endif
    int rc;

    // Defaults
    flags.target  = DEFAULT_TARGET;
    flags.dataset = DEFAULT_DATASET;
    flags.datadir = DEFAULT_DATADIR;
    flags.impl    = DEFAULT_IMPL;

    int opt;
    while ((opt = getopt_long_only(argc, argv, "t:o:rgaSi:h", long_options, NULL)) != -1)
    {
        switch (opt)
        {
        case OPT_TARGET:    flags.target     = optarg; break;
        case OPT_RELEASE:   flags.release    = true;   break;
        case OPT_DEBUG:     flags.debug      = true;   break;
        case OPT_ASAN:      flags.asan       = true;   break;
        case OPT_IMPL:
        {
            if      (strcmp("naive", optarg)  == 0) flags.impl = NAIVE_IMPL;
            else if (strcmp("blas", optarg)   == 0) flags.impl = BLAS_IMPL;
            else if (strcmp("tunded", optarg) == 0) flags.impl = TUNED_IMPL;
            else
            {
                nob_log(NOB_ERROR, "Invalid implentation was given: %s", optarg);
                usage(argv[0]);
                rc = ERR;
                goto exit;
            }
            break;
        }
        case OPT_OMP_OFF:   flags.omp_off    = true;   break;
        case OPT_ASM:       flags.asm_output = true;   break;
        case OPT_DBL:       flags.use_dbl    = true;   break;
        case OPT_OUTDIR:    flags.out_dir    = optarg; break;
        case OPT_MACRO:     nob_da_append(&flags.macros, optarg);     break;
        case OPT_RUN:       flags.run        = true;   break;
        case OPT_SLURM:     flags.slurm      = true;   break;
        case OPT_PARTITION: nob_da_append(&flags.partitions, optarg); break;
        case OPT_SCRIPT:    flags.script     = optarg; break;
        case OPT_DATASET:   flags.dataset    = optarg; break;
        case OPT_DATADIR:   flags.datadir    = optarg; break;
        case OPT_ETAGS:     flags.etags      = true;   break;
        case OPT_CLEAN:     flags.clean      = true;   break;
        case OPT_HELP:
            usage(argv[0]);
            rc = OK;
            goto exit;
        default:
            usage(argv[0]);
            rc = ERR;
            goto exit;
        }
    }

    flags.rest_argc = argc - optind;
    flags.rest_argv = argv + optind;

    flags.datadir = nob_expand_path(flags.datadir);

    // Default to debug if neither specified
    if (!flags.release && !flags.debug) {
        flags.debug = true;
    }

    // Default to debug if neither specified
    if (!flags.release && !flags.debug)
    {
        flags.debug = true;
    }

    if (strcmp(flags.target, "list") == 0)
    {
        list_targets();
        rc = OK;
    }

    if (flags.partitions.count > 0 && strcmp(flags.partitions.items[0], "list") == 0)
    {
        list_partitions();
        rc = OK;
    }

    // Commands
    if (flags.clean) {rc = clean();}
    else if (flags.etags) {rc = etags();}
    else
    {
        rc = prepare_dataset();
        if (rc == OK)
        {
            if (flags.slurm) {rc = slurm_via_worktree();}
            else {rc = build(flags.target, flags.run);}
        }
    }

    if (rc == ERR_USAGE) usage(argv[0]);

exit:
    return rc != OK;  // exit 0 or 1 to the shell
}
