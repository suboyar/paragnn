#define NOB_EXPERIMENTAL_DELETE_OLD
#define NOB_IMPLEMENTATION
#define NOB_WARN_DEPRECATED
#define nob_cc(cmd) nob_cmd_append(cmd, "gcc")
#include "nob.h"
#undef nob_cc_flags

#define FLAG_IMPLEMENTATION
#define FLAG_PUSH_DASH_DASH_BACK
#include "flag.h"

#define BUILD_FOLDER "build/"
#define TOOLS_FOLDER "tools/"
#define SRC_FOLDER "src/"

#define NOB_NEEDS_REBUILD(output_path, input_paths, input_paths_count) \
    (flags.rebuild ? 1 : nob_needs_rebuild((output_path), (input_paths), (input_paths_count)))

#define NOB_NEEDS_REBUILD1(output_path, input_path) \
    (flags.rebuild ? 1 : nob_needs_rebuild1((output_path), (input_path)))

#define nob_cc_flags(cmd) nob_cmd_append(cmd, "-std=c17", "-D_POSIX_C_SOURCE=200809L")
#define nob_cc_error_flags(cmd) \
    nob_cmd_append(cmd,                                                 \
                   "-Wall",                                             \
                   "-Wextra",                                           \
                   "-Wfloat-conversion",                                \
                   "-Werror=implicit-function-declaration",             \
                   "-Werror=incompatible-pointer-types")

#define SUBMIT_DESC "Job submission partition\n        Valid: defq, fpgaq, genoaxq, gh200q, milanq, xeonmaxq, rome16q, bench, dev, list"

typedef struct {
    const char* obj_path;
    const char* src_path;
    const char* exec_path;
} BuildTarget;

typedef struct {
    bool  build;
    bool  run;
    bool  rebuild;
    bool  release;
    bool  ogb;
    bool  help;
    char*  submit;
    char* out_dir;
    char* kernel;
    bool fastmath;
    Flag_List_Mut macros;
    int rest_argc;
    char** rest_argv;
} Flags;

Flags flags = {0};

typedef enum {
    CMD_NONE,
    CMD_CLEAN,
    CMD_OGB,
    CMD_COMPILE_LOCAL,
    CMD_COMPILE_SLURM,
    CMD_RUN_LOCAL,
    CMD_RUN_SLURM,
} Command;

Nob_Cmd cmd = {0};
Nob_Procs procs = {0};

struct {
    const char* name;
    const char* arch;
    bool dev;
} partitions[] = {
    // For benchmarking
    {.name = "defq",     .arch = "arm",   .dev = false},
    {.name = "fpgaq",    .arch = "amd",   .dev = false},
    {.name = "genoaxq",  .arch = "amd",   .dev = false},
    {.name = "gh200q",   .arch = "arm",   .dev = false},
    {.name = "milanq",   .arch = "amd",   .dev = false},
    {.name = "xeonmaxq", .arch = "intel", .dev = false},
    // For devlopment
    {.name = "rome16q",  .arch = "amd",   .dev = true},
};

bool nob_mkdir_if_not_exists_recursvily(const char *path)
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
        const char* dir_path = sb.items;

        if (!nob_mkdir_if_not_exists(dir_path)) return false;
    }
}

void list_all_partitions()
{
    // Find longest name
    int max_len = 0;
    for (size_t i = 0; i < NOB_ARRAY_LEN(partitions); i++) {
        int len = (int)strlen(partitions[i].name);
        if (len > max_len) max_len = len;
    }
    max_len += 2;    // Some more padding

    printf("Partitions for benchmarking:\n");
    for (size_t i = 0; i < NOB_ARRAY_LEN(partitions); i++) {
        if (!partitions[i].dev) {
            printf("\t%-*s (%s)\n", max_len, partitions[i].name, partitions[i].arch);
        }
    }

    printf("Partitions for testing:\n");
    for (size_t i = 0; i < NOB_ARRAY_LEN(partitions); i++) {
        if (partitions[i].dev) {
            printf("\t%-*s (%s)\n", max_len, partitions[i].name, partitions[i].arch);
        }
    }

    printf("\nExample usage: ./nob --sbatch rome16q\n");
}

bool partition_is_valid(const char* partition)
{
    for (size_t i = 0; i < NOB_ARRAY_LEN(partitions); i++) {
        if (strcmp(partitions[i].name, partition) == 0) return true;
    }

    return false;
}

char* to_define_flag(const char* str)
{
    Nob_String_Builder sb = {0};
    const char* p = str;

    while (*p) {
        nob_da_append(&sb, (char)toupper((unsigned char) *p++));
    }
    nob_sb_append_null(&sb);

    return nob_temp_sprintf("-D%s", sb.items);
}

int compile_src_files(BuildTarget* targets, size_t len)
{
    for (size_t i = 0; i < len; ++i) {
        if (NOB_NEEDS_REBUILD1(targets[i].obj_path, targets[i].src_path) > 0) {
            nob_cc(&cmd);
            nob_cc_flags(&cmd);
            nob_cc_error_flags(&cmd);
            nob_cmd_append(&cmd, "-I.");
            nob_cmd_append(&cmd, "-I"SRC_FOLDER);

            if (flags.release) {
                nob_cmd_append(&cmd, "-O3", "-march=native", "-DNDEBUG");
                if (flags.kernel == NULL) {
                    nob_cmd_append(&cmd, "-DUSE_OGB_ARXIV");
                    nob_cmd_append(&cmd, "-DEPOCH=10");
                }
            } else {
                nob_cmd_append(&cmd, "-O0", "-ggdb", "-g3", "-gdwarf-2");
            }

            for (size_t i = 0; i < flags.macros.count; ++i) {
                nob_cmd_append(&cmd, nob_temp_sprintf("-D%s", flags.macros.items[i]));
            }

            nob_cmd_append(&cmd, "-fopenmp");

            nob_cc_output(&cmd, targets[i].obj_path);
            nob_cmd_append(&cmd, "-c", targets[i].src_path);
            if (!nob_cmd_run(&cmd, .async = &procs)) return 1;
        }
    }

    if (!nob_procs_flush(&procs)) return 1;
}

int build_kernel_bench()
{
    char *out_dir = BUILD_FOLDER;
    char *kernel_dir = "kernels/";

    nob_mkdir_if_not_exists_recursvily(out_dir);

    const char* exec_path = NULL;
    const char* kernel_src = NULL;
    const char* kernel_obj = NULL;

    if (strcmp(flags.kernel, "dot-ex") == 0) {
        exec_path = nob_temp_sprintf("%s%s", out_dir, "dot-ex");
        kernel_src = nob_temp_sprintf("%s%s", kernel_dir, "dot_ex.c");
        kernel_obj = nob_temp_sprintf("%s%s", out_dir, "dot_ex.o");
    } else {
        nob_log(NOB_ERROR, "Wrong value passed to '-kernel': %s", flags.kernel);
        return 1;
    }

    BuildTarget targets[] = {
        {.obj_path = (nob_temp_sprintf("%s%s", out_dir, "matrix.o")), .src_path = SRC_FOLDER"matrix.c"},
        {.obj_path = (nob_temp_sprintf("%s%s", out_dir, "perf.o")),   .src_path = SRC_FOLDER"perf.c"},
        {.obj_path = kernel_obj,                                      .src_path = kernel_src}
    };

    // Compile src files
    if (compile_src_files(targets, NOB_ARRAY_LEN(targets)) != 0) return 1;

    const char* deps[NOB_ARRAY_LEN(targets)];
    for (size_t i = 0; i < NOB_ARRAY_LEN(targets); ++i) {
        deps[i] = targets[i].obj_path;
    }

    // Link object files
    if (NOB_NEEDS_REBUILD(exec_path, deps, NOB_ARRAY_LEN(deps)) > 0) {
        nob_cc(&cmd);
        nob_cc_flags(&cmd);
        nob_cc_error_flags(&cmd);
        nob_cmd_append(&cmd, "-I.");
        nob_cmd_append(&cmd, "-I"SRC_FOLDER);
        if (flags.release) {
            nob_cmd_append(&cmd, "-O3", "-march=native", "-DNDEBUG");
        } else {
            nob_cmd_append(&cmd, "-O0", "-ggdb", "-g3", "-gdwarf-2");
        }
        nob_cmd_append(&cmd, "-fopenmp");
        nob_cmd_append(&cmd, "-ffast-math");
        nob_cc_output(&cmd, exec_path);
        // nob_cc_inputs(&cmd, kernel_path);
        for (size_t i = 0; i < NOB_ARRAY_LEN(targets); ++i) {
            nob_cc_inputs(&cmd, targets[i].obj_path);
        }
        nob_cmd_append(&cmd, "-lm");
        if (!nob_cmd_run(&cmd)) return 1;
    }

    if (flags.run) {
        nob_cmd_append(&cmd, exec_path);
        if (!nob_cmd_run(&cmd)) return 1;
    }

    return 0;
}

int build_paragnn()
{
    const char *exec = "paragnn";
    char *out_dir = BUILD_FOLDER;
    if (flags.out_dir != NULL) {
        out_dir = flags.out_dir;
    }

    if ((out_dir[strlen(out_dir)-1]) != '/') {
        out_dir = nob_temp_sprintf("%s/", out_dir);
    }

    nob_mkdir_if_not_exists_recursvily(out_dir);

    const char* exec_path = nob_temp_sprintf("%s%s", out_dir, exec);

    BuildTarget targets[] = {
        {.obj_path = (nob_temp_sprintf("%s%s", out_dir, "gnn.o")),    .src_path = SRC_FOLDER"gnn.c"},
        {.obj_path = (nob_temp_sprintf("%s%s", out_dir, "graph.o")),  .src_path = SRC_FOLDER"graph.c"},
        {.obj_path = (nob_temp_sprintf("%s%s", out_dir, "layers.o")), .src_path = SRC_FOLDER"layers.c"},
        {.obj_path = (nob_temp_sprintf("%s%s", out_dir, "main.o")),   .src_path = SRC_FOLDER"main.c"},
        {.obj_path = (nob_temp_sprintf("%s%s", out_dir, "matrix.o")), .src_path = SRC_FOLDER"matrix.c"},
        {.obj_path = (nob_temp_sprintf("%s%s", out_dir, "perf.o")),   .src_path = SRC_FOLDER"perf.c"},
    };

    // Compile src files
    for (size_t i = 0; i < NOB_ARRAY_LEN(targets); ++i) {
        if (NOB_NEEDS_REBUILD1(targets[i].obj_path, targets[i].src_path) > 0) {
            nob_cc(&cmd);
            nob_cc_flags(&cmd);
            nob_cc_error_flags(&cmd);
            nob_cmd_append(&cmd, "-I.");
            nob_cmd_append(&cmd, "-I"SRC_FOLDER);

            if (flags.release) {
                nob_cmd_append(&cmd, "-O3", "-march=native", "-DNDEBUG");
                nob_cmd_append(&cmd, "-DUSE_OGB_ARXIV");
            } else {
                nob_cmd_append(&cmd, "-O0", "-ggdb", "-g3", "-gdwarf-2");
            }

            nob_cmd_append(&cmd, "-fopenmp");

            nob_cc_output(&cmd, targets[i].obj_path);
            nob_cmd_append(&cmd, "-c", targets[i].src_path);
            if (!nob_cmd_run(&cmd, .async = &procs)) return 1;
        }
    }

    if (!nob_procs_flush(&procs)) return 1;

    // Link object files
    const char* obj_paths[NOB_ARRAY_LEN(targets)];
    for (size_t i = 0; i < NOB_ARRAY_LEN(targets); ++i) {
        obj_paths[i] = targets[i].obj_path;
    }

    if (NOB_NEEDS_REBUILD(exec_path, obj_paths, NOB_ARRAY_LEN(targets)) > 0) {
        nob_cc(&cmd);
        nob_cc_flags(&cmd);
        nob_cc_error_flags(&cmd);
        if (flags.release) {
            nob_cmd_append(&cmd, "-O3", "-march=native");
        } else {
            nob_cmd_append(&cmd, "-O0", "-ggdb", "-g3", "-gdwarf-2");
        }
        nob_cmd_append(&cmd, "-fopenmp");
        nob_cc_output(&cmd, exec_path);
        for (size_t i = 0; i < NOB_ARRAY_LEN(targets); ++i) {
            nob_cc_inputs(&cmd, targets[i].obj_path);
        }
        nob_cmd_append(&cmd, "-lm");
        if (!nob_cmd_run(&cmd)) return 1;
    }

    if (flags.run) {
        nob_cmd_append(&cmd, exec_path);
        for (int i = 0; i < flags.rest_argc; i++) {
            nob_cmd_append(&cmd, flags.rest_argv[i]);
        }
        if (!nob_cmd_run(&cmd)) return 1;
    }

    return 0;
}


int build_ogb(){
    const char* exec = TOOLS_FOLDER"ogb";
    const char* obj_path = TOOLS_FOLDER"ogb.o";
    const char* src_path = TOOLS_FOLDER"ogb.c";

    // Compile src files
    if (NOB_NEEDS_REBUILD1(obj_path, src_path) > 0) {
        nob_cc(&cmd);
        nob_cc_flags(&cmd);
        nob_cc_error_flags(&cmd);
        nob_cmd_append(&cmd, "-I.");
        nob_cmd_append(&cmd, "-I"SRC_FOLDER);
        nob_cc_output(&cmd, obj_path);
        nob_cmd_append(&cmd, "-c", src_path);
        if (!nob_cmd_run(&cmd)) return 1;
    }

    // Link object files
    if (NOB_NEEDS_REBUILD1(exec, obj_path) > 0) {
        nob_cc(&cmd);
        nob_cc_flags(&cmd);
        nob_cc_error_flags(&cmd);
        nob_cmd_append(&cmd, "-O3");
        nob_cc_output(&cmd, exec);
        nob_cc_inputs(&cmd, obj_path);
        nob_cmd_append(&cmd, "-lz");
        if (!nob_cmd_run(&cmd)) return 1;
    }

    return 0;
}

int clean()
{
    nob_delete_file(BUILD_FOLDER);
    nob_delete_file(TOOLS_FOLDER"ogb");
    nob_delete_file(TOOLS_FOLDER"ogb.o");
    return 0;
}

int run_benchmark(const char* partition)
{
    nob_cmd_append(&cmd, "sbatch");
    nob_cmd_append(&cmd, "-p", partition);
    if (flags.kernel != NULL) {
        nob_cmd_append(&cmd, nob_temp_sprintf("--export=BENCHMARK_TARGET=%s", flags.kernel));
        nob_cmd_append(&cmd, "-J", flags.kernel);
    }
    nob_cmd_append(&cmd, "-o", nob_temp_sprintf("logs/%s-%%x-%%j.out", partition));
    nob_cmd_append(&cmd, "run_benchmark.sbatch");
    if (!nob_cmd_run(&cmd)) return 1;
    return 0;
}

int submit_job()
{
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
    } else {
        list_all_partitions();
    }

    return 1;
}

void usage(FILE *stream)
{
    fprintf(stream, "Usage: ./example [OPTIONS] [--] [ARGS]\n");
    fprintf(stream, "OPTIONS:\n");
    flag_print_options(stream);
}

int main(int argc, char** argv)
{
    NOB_GO_REBUILD_URSELF(argc, argv);

    flag_bool_var(&flags.build,     "build",    false, "Build the project");
    flag_bool_var(&flags.run,       "run",      false, "Run the project");
    flag_bool_var(&flags.rebuild,   "rebuild",  false, "Force a complete rebuild");
    flag_str_var(&flags.out_dir,    "outdir",   NULL,  "Where to build to");
    flag_bool_var(&flags.release,   "release",  false, "Build in release mode");
    flag_str_var(&flags.submit,     "submit",   NULL,  SUBMIT_DESC);
    flag_bool_var(&flags.ogb,       "ogb",      false, "Build OGB decoder");
    flag_str_var(&flags.kernel,     "kernel",   NULL,  "Build kernel benchmarks (dot_ex)");
    flag_bool_var(&flags.simd,      "simd",     false, "Enable simd");
    flag_bool_var(&flags.fastmath,  "fastmath", false, "Enable fastmath");
    flag_list_mut_var(&flags.macros, "D",               "Enable macro to be passed down (SIMD_ENABLED, USE_OGB_ARXIV, EPOCH)");
    flag_bool_var(&flags.help,      "help",     false, "Print this help message");

    if (!flag_parse(argc, argv)) {
        usage(stderr);
        flag_print_error(stderr);
        return 1;
    }

    if (flags.help) {
        usage(stdout);
        return 0;
    }

    argc = flag_rest_argc();
    argv = flag_rest_argv();

    if (argc > 0 && strcmp(argv[0], "--") == 0) {
        argv += 1;
        argc -= 1;
    }

    flags.rest_argc = argc;
    flags.rest_argv = argv;

    if (flags.ogb) {
        return build_ogb();
    }

    if (flags.submit != NULL) {
        if (strcmp(flags.submit, "list") == 0) {
            list_all_partitions();
            return 0;
        }

        return submit_job();
    }

    if (flags.kernel != NULL) {
        return build_kernel_bench();
    }

    if (flags.build || flags.rebuild) {
        return build_paragnn();
    }

    return 0;
}
