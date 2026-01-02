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
    bool   extract_ogb;
    bool   download_ogb;
    bool   help;
    char*  target;
    char*  submit;
    char*  out_dir;
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

void append_common_flags(bool debug, bool release)
{
    nob_cc_flags(&cmd);
    nob_cc_error_flags(&cmd);
    nob_cmd_append(&cmd, "-I"SRC_FOLDER);

    if (debug || !release) {
        nob_cmd_append(&cmd, "-ggdb", "-g3", "-gdwarf-2");
    }

    if (release) {
        nob_cmd_append(&cmd, "-O3", "-march=native", "-DNDEBUG", "-ffast-math");
    } else {
        nob_cmd_append(&cmd, "-O0");
    }

    nob_cmd_append(&cmd, "-fopenmp");
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
        append_common_flags(debug, release);

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
    append_common_flags(debug, release);

    nob_cc_output(&cmd, exec_path);

    for (size_t i = 0; i < t->srcs.count; i++) {
        nob_cc_inputs(&cmd, obj_paths[i]);
    }

    for (size_t i = 0; i < t->libs.count; i++) {
        nob_cmd_append(&cmd, t->libs.items[i]);
    }

    if (!nob_cmd_run(&cmd)) return 1;

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

struct {
    const char* name;
    const char* arch;
    bool dev;
} partitions[] = {
    {.name = "defq",     .arch = "arm",   .dev = false},
    {.name = "fpgaq",    .arch = "amd",   .dev = false},
    {.name = "genoaxq",  .arch = "amd",   .dev = false},
    {.name = "gh200q",   .arch = "arm",   .dev = false},
    {.name = "milanq",   .arch = "amd",   .dev = false},
    {.name = "xeonmaxq", .arch = "intel", .dev = false},
    {.name = "rome16q",  .arch = "amd",   .dev = true},
};

void list_partitions(void)
{
    int max_len = 0;
    for (size_t i = 0; i < NOB_ARRAY_LEN(partitions); i++) {
        int len = (int)strlen(partitions[i].name);
        if (len > max_len) max_len = len;
    }
    max_len += 2;

    printf("Partitions for benchmarking:\n");
    for (size_t i = 0; i < NOB_ARRAY_LEN(partitions); i++) {
        if (!partitions[i].dev) {
            printf("  %-*s (%s)\n", max_len, partitions[i].name, partitions[i].arch);
        }
    }

    printf("Partitions for development:\n");
    for (size_t i = 0; i < NOB_ARRAY_LEN(partitions); i++) {
        if (partitions[i].dev) {
            printf("  %-*s (%s)\n", max_len, partitions[i].name, partitions[i].arch);
        }
    }
}

bool partition_is_valid(const char* name)
{
    for (size_t i = 0; i < NOB_ARRAY_LEN(partitions); i++) {
        if (strcmp(partitions[i].name, name) == 0) return true;
    }
    return false;
}

int run_benchmark(const char* partition)
{
    nob_cmd_append(&cmd, "sbatch", "-p", partition);

    if (flags.target && strcmp(flags.target, "paragnn") != 0) {
        nob_cmd_append(&cmd, nob_temp_sprintf("--export=BENCHMARK_TARGET=%s", flags.target));
        nob_cmd_append(&cmd, "-J", flags.target);
    }

    nob_cmd_append(&cmd, "-o", nob_temp_sprintf("logs/%s-%%x-%%j.out", partition));
    nob_cmd_append(&cmd, "run_benchmark.sbatch");

    return nob_cmd_run(&cmd) ? 0 : 1;
}

int submit_job(void)
{
#if 0
    if (!nob_mkdir_if_not_exists("logs/")) return 1;

    bool is_bench = strcmp(flags.submit, "bench") == 0;
    bool is_dev = strcmp(flags.submit, "dev") == 0;

    if (is_bench || is_dev) {
        for (size_t i = 0; i < NOB_ARRAY_LEN(partitions); i++) {
            if (partitions[i].dev != is_dev) continue;
            if (run_benchmark(partitions[i].name)) return 1;
        }
        return 0;
    }

    if (partition_is_valid(flags.submit)) {
        return run_benchmark(flags.submit);
    }

    nob_log(NOB_ERROR, "Unknown partition: %s", flags.submit);
    list_partitions();
    return 1;
#else
    NOB_TODO("submitting needs to be redone");
    return 1;
#endif
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
    NOB_GO_REBUILD_URSELF(argc, argv);

    flag_str_var(&flags.target,        "target",       "paragnn", "Build target (see list below)");
    flag_bool_var(&flags.download_ogb, "download-ogb", false,     "Download OGB arxiv dataset");
    flag_bool_var(&flags.extract_ogb,  "extract-ogb",  false,     "Re-extract OGB raw files");
    flag_bool_var(&flags.run,          "run",          false,     "Run after building");
    flag_bool_var(&flags.debug,        "debug",        false,     "Build in debug mode (default)");
    flag_bool_var(&flags.release,      "release",      false,     "Build in release mode");
    flag_str_var(&flags.out_dir,       "outdir",       NULL,      "Output directory");
    flag_str_var(&flags.submit,        "submit",       NULL,      "Submit to Slurm (partition or 'bench'/'dev'/'list')");
    flag_bool_var(&flags.clean,        "clean",        false,     "Clean build artifacts");
    flag_list_mut_var(&flags.macros,   "D",                       "Define macro (e.g., -D SIMD_ENABLED)");
    flag_bool_var(&flags.help,         "help",         false,     "Print this help message");

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

    if (flags.submit != NULL) {
        if (strcmp(flags.submit, "list") == 0) {
            list_partitions();
            return 0;
        }
        return submit_job();
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
