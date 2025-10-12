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

typedef struct {
    bool  build;
    bool  run;
    bool  rebuild;
    bool  build_ogb;
    bool  release;
    bool  submit;
    char* outdir;
    char* partition;
    char* variant;
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
    bool testing;
} partitions[] = {
    // For benchmarking
    {.name = "defq",     .arch = "arm",   .testing = false},
    {.name = "fpgaq",    .arch = "amd",   .testing = false},
    {.name = "genoaxq",  .arch = "amd",   .testing = false},
    {.name = "gh200q",   .arch = "arm",   .testing = false},
    {.name = "milanq",   .arch = "amd",   .testing = false},
    {.name = "xeonmaxq", .arch = "intel", .testing = false},
    // For testing
    {.name = "rome16q",  .arch = "amd",   .testing = true},
};

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
        if (!partitions[i].testing) {
            printf("\t%-*s (%s)\n", max_len, partitions[i].name, partitions[i].arch);
        }
    }

    printf("Partitions for testing:\n");
    for (size_t i = 0; i < NOB_ARRAY_LEN(partitions); i++) {
        if (partitions[i].testing) {
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

int run_paragnn()
{
    const char *exec = "paragnn";
    char *out_dir = BUILD_FOLDER;

    if (flags.variant != NULL){
        out_dir = nob_temp_sprintf("%s%s/", out_dir, flags.variant);
    }

    // if (flags.partition != NULL){
    //     out_dir = nob_temp_sprintf("%s%s/", out_dir, flags.partition);
    //     if (!nob_mkdir_if_not_exists(out_dir)) return 1;
    // }

    const char* exec_path = nob_temp_sprintf("./%s%s", out_dir, exec);
    int ret = nob_file_exists(exec_path);
    if (ret != 1) {
        if (ret == 0) {
            nob_log(NOB_ERROR, "Failed to execute %s: executable not found", exec_path);
        }
        return 1;
    }

    nob_cmd_append(&cmd, exec_path);
    if (!nob_cmd_run(&cmd)) return 1;


    return 0;
}

int build_paragnn()
{
    const char *exec = "paragnn";
    char *out_dir = BUILD_FOLDER;
    if (!nob_mkdir_if_not_exists(out_dir)) return 1;

    if (flags.variant != NULL){
        out_dir = nob_temp_sprintf("%s%s/", out_dir, flags.variant);
        if (!nob_mkdir_if_not_exists(out_dir)) return 1;
    }

    if (flags.partition != NULL){
        out_dir = nob_temp_sprintf("%s%s/", out_dir, flags.partition);
        if (!nob_mkdir_if_not_exists(out_dir)) return 1;
    }

    const char* exec_path = nob_temp_sprintf("%s%s", out_dir, exec);
    const char* variant_path = SRC_FOLDER;
    if (flags.variant != NULL) {
        variant_path = nob_temp_sprintf("%s%s/", variant_path, flags.variant);
    }
    struct {
        const char* obj_path;
        const char* src_path;
    } targets[] = {
        {.obj_path = (nob_temp_sprintf("%s%s", out_dir, "gnn.o")),    .src_path = nob_temp_sprintf("%s%s", variant_path, "gnn.c")},
        {.obj_path = (nob_temp_sprintf("%s%s", out_dir, "graph.o")),  .src_path = SRC_FOLDER"graph.c"},
        {.obj_path = (nob_temp_sprintf("%s%s", out_dir, "layers.o")), .src_path = nob_temp_sprintf("%s%s", variant_path, "layers.c")},
        {.obj_path = (nob_temp_sprintf("%s%s", out_dir, "main.o")),   .src_path = SRC_FOLDER"main.c"},
        {.obj_path = (nob_temp_sprintf("%s%s", out_dir, "matrix.o")), .src_path = nob_temp_sprintf("%s%s", variant_path, "matrix.c")},
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
	        nob_cmd_append(&cmd, nob_temp_sprintf("-I%s", variant_path));

            if (flags.variant != NULL) {
                if (strcmp(flags.variant, "baseline") == 0) {
                    // Default optimizations
                    nob_cmd_append(&cmd, "-DBASELINE");
                }
            }

            if (flags.release) {
                nob_cmd_append(&cmd, "-O3", "-march=native", "-DNDEBUG");
                nob_cmd_append(&cmd, "-DUSE_OGB_ARXIV");
            } else {
                nob_cmd_append(&cmd, "-O0", "-ggdb", "-g3", "-gdwarf-2");
            }

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

int submit_job()
{
    if (flags.build) {
        if (!nob_mkdir_if_not_exists("./logs/compile")) return 1;

        nob_cmd_append(&cmd, "sbatch");
        nob_cmd_append(&cmd, "-p", flags.partition);
        nob_cmd_append(&cmd, "--export", nob_temp_sprintf("ALL,EXPERIMENT=%s", flags.variant));
        nob_cmd_append(&cmd, "ex3/compile.sbatch");
        if (!nob_cmd_run(&cmd)) return 1;
    } else if (flags.run) {
        nob_cmd_append(&cmd, "sbatch");
        nob_cmd_append(&cmd, "-p", flags.partition);
        nob_cmd_append(&cmd, "--export", nob_temp_sprintf("ALL,EXPERIMENT=%s", flags.variant));
        nob_cmd_append(&cmd, nob_temp_sprintf("%s-run.sbatch", flags.partition));
    } else {
        nob_log(NOB_ERROR, "No job valid job");
    }

    return 0;
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

    flag_bool_var(&flags.build,         "build",     false,      "Build the project");
    flag_bool_var(&flags.run,           "run",       false,      "Run the project");
    flag_bool_var(&flags.rebuild,       "rebuild",   false,      "Force a complete rebuild");
    flag_bool_var(&flags.build_ogb,     "build-ogb", false,      "Build OGB decoder");
    flag_bool_var(&flags.release,       "release",   false,      "Build in release mode");
    flag_bool_var(&flags.submit,        "submit",    false,      "Submit build job to SLURM instead of building locally");
    flag_str_var(&flags.partition,      "partition", NULL,       "SLURM partition to use");
    flag_str_var(&flags.variant,        "variant",   NULL, "Build variant (baseline, etc.)");

    if (!flag_parse(argc, argv)) {
        usage(stderr);
        flag_print_error(stderr);
        exit(1);
    }

    if (flags.build_ogb) {
        return build_ogb();
    }

    else if (flags.submit) {
        return submit_job();
    }

    if (flags.run) {
        if (flags.build || flags.rebuild) {
            if (build_paragnn()) return 1;
        }
        return run_paragnn();
    }

    if (flags.build || flags.rebuild) {
        return build_paragnn();
    }

    return 0;
}
