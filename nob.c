#define NOB_EXPERIMENTAL_DELETE_OLD
#define NOB_IMPLEMENTATION
#define NOB_WARN_DEPRECATED
#define NOB_REBUILD_URSELF(binary_path, source_path) "cc", "-ggdb", "-o", binary_path, source_path
#define nob_cc(cmd) nob_cmd_append(cmd, "gcc")
#include "nob.h"
#undef nob_cc_flags

#define FLAG_IMPLEMENTATION
#define FLAG_PUSH_DASH_DASH_BACK
#include "flag.h"

#define BUILD_FOLDER "build/"
#define SRC_FOLDER   "src/"
#define KERNEL_FOLDER "kernels/"

#define OGB_ARXIV_PATH "./arxiv/"
#define OGB_ARXIV_URL "http://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip"
#define OGB_ARXIV_RAW OGB_ARXIV_PATH"raw/"
#define OGB_ARXIV_PROCESSED OGB_ARXIV_PATH"processed/"
#define OGB_ARXIV_FILES_COUNT 6


#define nob_cc_flags(cmd) nob_cmd_append(cmd, "-std=c17", "-D_POSIX_C_SOURCE=200809L")
#define nob_cc_error_flags(cmd) \
    nob_cmd_append(cmd,         \
        "-Wall",                \
        "-Wextra",              \
        "-Wfloat-conversion",   \
        "-Werror=implicit-function-declaration", \
        "-Werror=incompatible-pointer-types") // Maybe re-add -Wno-unknown-pragmas?

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
            SRC_FOLDER"gnn.c",
            SRC_FOLDER"graph.c",
            SRC_FOLDER"layers.c",
            SRC_FOLDER"matrix.c",
            SRC_FOLDER"linalg/axpy.c",
            SRC_FOLDER"linalg/copy.c",
            SRC_FOLDER"linalg/scal.c",
            SRC_FOLDER"linalg/gemm.c",
            SRC_FOLDER"timer.c",
            ),
        .libs = STRINGS("-lm", "-lopenblas"),
        .release_macros = STRINGS("-DUSE_OGB_ARXIV"),
    },
    {
        .name = "tsmm_tn",
        .srcs = STRINGS(
            SRC_FOLDER"matrix.c",
            SRC_FOLDER"timer.c",
            KERNEL_FOLDER"cache_counter.c",
            KERNEL_FOLDER"tsmm_tn.c",
            SRC_FOLDER"linalg/gemm.c",
            ),
        .libs = STRINGS("-lm", "-lopenblas"),
    },
};

typedef struct {
    bool   run;
    bool   debug;
    bool   release;
    bool   clean;
    bool   omp_off;
    bool   extract_ogb;
    bool   download_ogb;
    bool   help;
    char*  target;
    char*  out_dir;
    bool   slurm;
    Flag_List_Mut partitions;
    Flag_List_Mut macros;
    int    rest_argc;
    char** rest_argv;
} Flags;

Flags flags = {0};
Nob_Cmd cmd = {0};
Nob_Procs procs = {0};

size_t ptrlen(void **arr) {
    size_t n = 0;
    while (arr[n]) n++;
    return n;
}

bool mkdir_recursive(const char *path)
{
    if (path == NULL || path[0] == '\0') {
        nob_log(NOB_ERROR, "cannot create directory from empty path");
        return false;
    }

    Nob_String_View sv = nob_sv_from_cstr(path);
    Nob_String_Builder sb = {0};

    while (sv.count > 0) {
        Nob_String_View dir = nob_sv_chop_by_delim(&sv, '/');
        if (dir.count == 0) continue;

        nob_sb_appendf(&sb, "%s/", nob_temp_sv_to_cstr(dir));
        if (!nob_mkdir_if_not_exists(sb.items)) return false;
    }
    return true;
}

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

const char* get_obj_path(const char* out_dir, const char* src_path)
{
    const char* base = nob_path_name(src_path);
    Nob_String_View sv = nob_sv_from_cstr(base);

    // Strip .c extension
    size_t len = sv.count;
    if (len > 2 && sv.data[len-2] == '.' && sv.data[len-1] == 'c') {
        len -= 2;
    }

    return nob_temp_sprintf("%s%.*s.o", out_dir, (int)len, sv.data);
}

typedef enum {
    COMPILING,
    LINKING,
} BuildPhase;

void append_common_flags(bool release, BuildPhase phase)
{
    nob_cc_flags(&cmd);
    nob_cc_error_flags(&cmd);
    nob_cmd_append(&cmd, "-I"SRC_FOLDER);

    if (!release) {
        nob_cmd_append(&cmd, "-ggdb", "-g3", "-gdwarf-2");
    }

    if (release) {
        nob_cmd_append(&cmd, "-O3", "-march=native", "-DNDEBUG", "-ffast-math");
    } else {
        nob_cmd_append(&cmd, "-O0");
    }

    if (phase == COMPILING && flags.omp_off) {
        nob_cmd_append(&cmd, "-Wno-unknown-pragmas");
    } else {
        nob_cmd_append(&cmd, "-fopenmp");
    }
}

bool ungzip(Nob_File_Paths *file_paths, const char* subfolder)
{
    for (size_t i = 0; i < file_paths->count; i++) {
        const char *files = file_paths->items[i];
        if (*files == '.') continue;
        nob_log(NOB_INFO, "Extracting %s to %s", files, OGB_ARXIV_PROCESSED);

        if (!subfolder) {
            subfolder = ".";
        }

        char *in = nob_temp_sprintf(OGB_ARXIV_PATH"%s/%s", subfolder, files);

        size_t base_len = strlen(files) - 3; // strip .gz
        const char *out = nob_temp_sprintf("%s%.*s", OGB_ARXIV_PROCESSED, (int)base_len, files);

        nob_cmd_append(&cmd, "gzip", "-d", "-c", in);
        if (!nob_cmd_run(&cmd, .stdout_path = out)) {
            nob_log(NOB_ERROR, "Failed to extract %s", files);
            return false;
        }
    }
    return true;
}

#define OGB_ARXIV_PROCESSED_COUNT 9
bool prepare_ogb_dataset()
{
    bool result = true;
    Nob_File_Paths processed_file_paths = {0};
    Nob_File_Paths raw_file_paths = {0};
    Nob_File_Paths split_file_paths = {0};

    if (!flags.extract_ogb &&
        nob_read_entire_dir(OGB_ARXIV_PROCESSED, &processed_file_paths) &&
        processed_file_paths.count == OGB_ARXIV_PROCESSED_COUNT+2) { // . and .. included in dir listing
        nob_log(NOB_INFO, "OGB dataset already processed");
        nob_return_defer(true);
    }

    if(!nob_read_entire_dir(OGB_ARXIV_PATH"raw", &raw_file_paths) ||
       !nob_read_entire_dir(OGB_ARXIV_PATH"split/time", &split_file_paths)) {
        if (!flags.download_ogb) {
            // Maybe support downloading other datasets from OGB, than only supporting ogb-arxiv
            nob_log(NOB_ERROR, "OGB dataset not found. Run with -download-ogb to fetch it");
            nob_return_defer(false);
        }

        // TODO: try this later when I'm not on metered connection
        nob_log(NOB_INFO, "Downloading OGB arxiv dataset...");
        nob_cmd_append(&cmd, "wget", "-q", "--show-progress", OGB_ARXIV_URL);
        if (!nob_cmd_run(&cmd)) {
            nob_log(NOB_ERROR, "Failed to download ogb-arxiv dataset");
            nob_return_defer(false);
        }

        nob_cmd_append(&cmd, "unzip", "-q", "arxiv.zip"); // No need for -d OGB_ARXIV_PATH as the zip already has files under arxiv/
        if (!nob_cmd_run(&cmd)) {
            nob_log(NOB_ERROR, "Failed to unzip arxiv.zip");
            nob_return_defer(false);
        }

        // Try reading the files one more time
        if(!nob_read_entire_dir(OGB_ARXIV_PATH"raw", &raw_file_paths)) {
            nob_log(NOB_ERROR, "Failed to read raw files after extraction");
            nob_return_defer(false);
        }
        if(!nob_read_entire_dir(OGB_ARXIV_PATH"split/time", &split_file_paths)) {
            nob_log(NOB_ERROR, "Failed to read split files after extraction");
            nob_return_defer(false);
        }
    }

    if (!ungzip(&raw_file_paths, "raw")) nob_return_defer(false);
    if (!ungzip(&split_file_paths, "split/time")) nob_return_defer(false);

    nob_log(NOB_INFO, "OGB dataset ready");

defer:
    nob_da_free(processed_file_paths);
    nob_da_free(raw_file_paths);
    nob_da_free(split_file_paths);

    return result;
}

int build_target(Target* t)
{
    // Download-extract-decompress dataset if needed
    if (strcmp(t->name, "paragnn") == 0) {
        if (!prepare_ogb_dataset()) return 1;
    }

    bool release = flags.release;
    bool debug = flags.debug;

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
    const char** obj_paths = nob_temp_alloc(t->srcs.count * sizeof(char*));

    for (size_t i = 0; i < t->srcs.count; i++) {
        const char* src = t->srcs.items[i];
        const char* obj = get_obj_path(out_dir, src);
        obj_paths[i] = obj;

        nob_cc(&cmd);
        append_common_flags(release, COMPILING);

        if (release && t->release_macros.items) {
            for (size_t i = 0; i < t->release_macros.count; i++) {
                bool overwrite = false;
                for (size_t j = 0; j < flags.macros.count; j++) {
                    char *user_macro = nob_temp_sprintf("-D%s", flags.macros.items[j]);
                        if (strcmp(user_macro, t->release_macros.items[i]) == 0) {
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

        nob_cc_output(&cmd, obj);
        nob_cmd_append(&cmd, "-c", src);

        if (!nob_cmd_run(&cmd, .async = &procs)) {
            nob_log(NOB_ERROR, "Failed to compile %s", src);
            return 1;
        }
    }

    if (!nob_procs_flush(&procs)) return 1;

    // Link
    nob_cc(&cmd);
    append_common_flags(release, LINKING);
    nob_cmd_append(&cmd, "-fopenmp");

    nob_cc_output(&cmd, exec_path);

    for (size_t i = 0; i < t->srcs.count; i++) {
        nob_cc_inputs(&cmd, obj_paths[i]);
    }

    for (size_t i = 0; i < t->libs.count; i++) {
        nob_cmd_append(&cmd, t->libs.items[i]);
    }

    if (!nob_cmd_run(&cmd)) return 1;

    nob_log(NOB_INFO, "Succesfully compiled: %s", exec_path);

    // Run if requested
    if (flags.run) {
        nob_cmd_append(&cmd, exec_path);
        for (int i = 0; i < flags.rest_argc; i++) {
            nob_cmd_append(&cmd, flags.rest_argv[i]);
        }
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

int submit_slurm(void)
{
    if (!nob_mkdir_if_not_exists("logs/")) return 1;

    size_t count;
    const char **parts = resolve_partitions(&count);
    if (!parts) return 1;

    const char *config = get_config_suffix();

    Nob_String_Builder macros_str = {0};
    for (size_t i = 0; i < flags.macros.count; i++) {
        if (i > 0) nob_sb_append_cstr(&macros_str, " ");
        nob_sb_append_cstr(&macros_str, nob_temp_sprintf("-D %s", flags.macros.items[i]));
    }
    if (macros_str.count == 0) nob_sb_append_null(&macros_str);

    time_t now = time(NULL);
    struct tm *local_time = localtime(&now);
    char date_string[20];
    strftime(date_string, sizeof(date_string), "%Y%m%d-%H%M%S", local_time);

    int ret = 0;
    for (size_t i = 0; i < count; i++) {
        nob_cmd_append(&cmd, "sbatch", "-p", parts[i]);
        nob_cmd_append(&cmd, nob_temp_sprintf("--export=BENCHMARK_TARGET=%s,BENCHMARK_CONFIG=%s,BENCHMARK_MACROS=%s",
                                              flags.target, config, macros_str.items));
        nob_cmd_append(&cmd, "-J", nob_temp_sprintf("%s-%s", flags.target, config));
        nob_cmd_append(&cmd, "-o", nob_temp_sprintf("logs/%s-%s-%s-%s.out",
                                                    date_string, parts[i], flags.target, config));
        nob_cmd_append(&cmd, "--exclusive");
        nob_cmd_append(&cmd, "run_benchmark.sbatch");

        if (!nob_cmd_run(&cmd)) {
            nob_log(NOB_ERROR, "Submitting %s to %s", flags.target, parts[i]);
            ret = 1;
        } else {
            nob_log(NOB_INFO, "Submitting %s to %s", flags.target, parts[i]);
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

void usage(FILE* stream)
{
    fprintf(stream, "Usage: ./nob [OPTIONS] [--] [ARGS]\n\n");
    fprintf(stream, "OPTIONS:\n");
    flag_print_options(stream);
    fprintf(stream, "\nTARGETS:\n");
    list_targets();
}

int main(int argc, char** argv)
{
#ifndef NOB_NO_REBUILD
    NOB_GO_REBUILD_URSELF(argc, argv);
#endif

    flag_str_var(&flags.target,          "target",       "paragnn", "Build target (see list below)");
    flag_bool_var(&flags.download_ogb,   "download-ogb", false,     "Download OGB arxiv dataset");
    flag_bool_var(&flags.extract_ogb,    "extract-ogb",  false,     "Re-extract OGB raw files");
    flag_bool_var(&flags.run,            "run",          false,     "Run after building");
    flag_bool_var(&flags.slurm,          "slurm",        false,     "Submit job to Slurm");
    flag_list_mut_var(&flags.partitions, "p",                       "Partition(s) to submit to (or 'list', 'all', 'aarch64', 'x86_64')");
    flag_bool_var(&flags.debug,          "debug",        false,     "Build in debug mode (default)");
    flag_bool_var(&flags.omp_off,        "omp-off",      false,     "Don't compile with -fopenmp");
    flag_bool_var(&flags.release,        "release",      false,     "Build in release mode");
    flag_str_var(&flags.out_dir,         "o",            NULL,      "Output directory");
    flag_bool_var(&flags.clean,          "clean",        false,     "Clean build artifacts");
    flag_list_mut_var(&flags.macros,     "D",                       "Define macro (e.g., -D SIMD_ENABLED)");
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

    return build_target(t);
}
