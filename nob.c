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
    char*         datadir;

    bool          etags;
    bool          help;
    bool          clean;

    int           rest_argc;
    char**        rest_argv;
} Flags;

Flags flags = {0};
Nob_Cmd cmd = {0};
Nob_Procs procs = {0};

void list_datasets(void)
{
    printf("Available datasets:\n");
    for (size_t i = 0; i < NOB_ARRAY_LEN(ds_infos); i++)
    {
        printf("  %s\n", ds_infos[i].name);
    }
}

int prepare_dataset()
{
    char *bin_path = "./prepare_dataset";
    char *src_path = "./prepare_dataset.c";
    if (nob_needs_rebuild1(bin_path, src_path))
    {
        nob_log(NOB_INFO, "Building %s", nob_path_name(bin_path));
        nob_cc(&cmd);
        nob_cc_flags(&cmd);
        nob_cc_error_flags(&cmd);
        nob_cmd_append(&cmd, "-march=native", "-O3", "-fopenmp", "-lz");
        nob_cc_output(&cmd, bin_path);
        nob_cc_inputs(&cmd, src_path);
        if (!nob_cmd_run(&cmd)) return EXIT_FAILURE;
    }

    nob_cmd_append(&cmd, bin_path, "-dataset", flags.dataset, "-datadir", flags.datadir);
    if (!nob_cmd_run(&cmd)) return EXIT_FAILURE;
    return 1;
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
    return nob_temp_sprintf("%s%.*s%s", out_dir, (int)len, sv.data, ext); // TODO: use path_join?
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
    if (!nob_mkdir_recursive(out_dir)) return 1;
    const char* exec_path = nob_path_join_temp(out_dir, t->name);

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

        if (!nob_cmd_run
            (&cmd, .async = &procs)) {
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
        if (flags.datadir) nob_cmd_append(&cmd, "-datadir", flags.datadir);
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
    if (!nob_mkdir_recursive(log_dir)) return 1;

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
        nob_cmd_append(&cmd, "-ipath", nob_temp_sprintf("*/%s*", folders[i]));
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
    flag_str_var(&flags.datadir,         "datadir",      "./data",  "Directory for downloading and reading datasets");

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
