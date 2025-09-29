#define nob_cc_flags(cmd) nob_cmd_append(cmd, "-std=c17", "-D_POSIX_C_SOURCE=200809L")
#define nob_cc_error_flags(cmd) nob_cmd_append(cmd, "-Wall", \
                                                    "-Wextra", \
                                                    "-Wfloat-conversion", \
                                                    "-Werror=implicit-function-declaration", \
                                                    "-Werror=incompatible-pointer-types")
#define NOB_IMPLEMENTATION
#define NOB_WARN_DEPRECATED
#include "nob.h"

#define BUILD_FOLDER "build/"
#define TOOLS_FOLDER "tools/"
#define SRC_FOLDER "src/"

#define NOB_NEEDS_REBUILD(output_path, input_paths, input_paths_count) \
    (flags.force_rebuild ? 1 : nob_needs_rebuild((output_path), (input_paths), (input_paths_count)))

#define NOB_NEEDS_REBUILD1(output_path, input_path) \
    (flags.force_rebuild ? 1 : nob_needs_rebuild1((output_path), (input_path)))

static struct {
    bool        force_rebuild;
    bool        build_ogb;
    bool        release;
    bool        compile_sbatch;
    bool        run_sbatch;
    const char* partition;
    const char* out_dir;
} flags = {
    .force_rebuild  = false,
    .build_ogb      = false,
    .release        = false,
    .compile_sbatch = false,
    .run_sbatch     = false,
    .partition      = NULL,
    .out_dir        = NULL,
};

Nob_Cmd cmd = {0};
Nob_Procs procs = {0};

struct {
    const char* name;
    const char* arch;
    bool testing;
} partitions[] = {
    {.name = "defq",     .arch = "arm",   .testing = false},
    {.name = "genoaxq",  .arch = "amd",   .testing = false},
    {.name = "milanq",   .arch = "amd",   .testing = false},
    {.name = "fpgaq",    .arch = "amd",   .testing = false},
    {.name = "xeonmaxq", .arch = "intel", .testing = false},
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

int build_paragnn(const char* exec, const char* out_dir)
{
    const char* exec_path = nob_temp_sprintf("%s%s", out_dir, exec);
    struct {
        const char* obj_path;
        const char* src_path;
    } targets[] = {
        {.obj_path = (const char*)(nob_temp_sprintf("%s%s", out_dir, "gnn.o")),       .src_path = SRC_FOLDER"gnn.c"},
        {.obj_path = (const char*)(nob_temp_sprintf("%s%s", out_dir, "graph.o")),     .src_path = SRC_FOLDER"graph.c"},
        {.obj_path = (const char*)(nob_temp_sprintf("%s%s", out_dir, "layers.o")),    .src_path = SRC_FOLDER"layers.c"},
        {.obj_path = (const char*)(nob_temp_sprintf("%s%s", out_dir, "main.o")),      .src_path = SRC_FOLDER"main.c"},
        {.obj_path = (const char*)(nob_temp_sprintf("%s%s", out_dir, "matrix.o")),    .src_path = SRC_FOLDER"matrix.c"},
        {.obj_path = (const char*)(nob_temp_sprintf("%s%s", out_dir, "perf.o")), .src_path = SRC_FOLDER"perf.c"},
    };

    // Compile src files
    for (size_t i = 0; i < NOB_ARRAY_LEN(targets); ++i) {
        if (NOB_NEEDS_REBUILD1(targets[i].obj_path, targets[i].src_path) > 0) {
            nob_cc(&cmd);
            nob_cc_flags(&cmd);
            nob_cc_error_flags(&cmd);
            nob_cmd_append(&cmd, "-I.");
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


int build_ogb(const char* exec, const char* out_dir){
    const char* exec_path = nob_temp_sprintf("%s%s", out_dir, exec);
    const char* obj_path = nob_temp_sprintf("%s%s", out_dir, "ogb.o");
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
        nob_cc_output(&cmd, exec_path);
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

int sbatch(const char* partition)
{
    nob_cmd_append(&cmd, "sbatch");
    nob_cmd_append(&cmd, "-p", partition);
    if (flags.run_sbatch) nob_cmd_append(&cmd, "run.sbatch");
    else nob_cmd_append(&cmd, "compile.sbatch");

    if (!nob_cmd_run(&cmd)) return 1;

    return 0;
}

int main(int argc, char** argv)
{
    NOB_GO_REBUILD_URSELF(argc, argv);

    // Parse flags
    nob_shift(argv, argc);      // jump over executable name in argv[0]
    while (argc) {
        const char* arg = nob_shift(argv, argc);
        if (strcmp(arg, "--clean") == 0) {
            return clean();
        } else if (strcmp(arg, "--compile-sbatch") == 0) {
            if (argc == 0) {
                list_all_partitions();
                fprintf(stderr, "Error: --compile-sbatch requires an argument, it should be one of: [list,all,partition name...]\n");
                return 1;
            }

            if (flags.run_sbatch) {
                fprintf(stderr, "Error: --compile-sbatch can not be called together with --run-sbatch\n");
                return 1;
            }
            flags.compile_sbatch = true;
            flags.partition = nob_shift(argv, argc);
            if (strcmp(flags.partition, "list") == 0) {
                list_all_partitions();
                return 0;
            } else if (strcmp(flags.partition, "all") != 0 && !partition_is_valid(flags.partition)) {
                list_all_partitions();
                fprintf(stderr,
                        "Error: --compile-sbatch received invalid argument '%s', it should be one of: [list,all,partition name...]\n",
                        flags.partition);
                return 1;
            }
        } else if (strcmp(arg, "--run-sbatch") == 0) {
            if (argc == 0) {
                list_all_partitions();
                fprintf(stderr, "Error: --run-sbatch requires an argument, it should be one of: [list,all,partition name...]\n");
                return 1;
            }

            if (flags.compile_sbatch) {
                fprintf(stderr, "Error: --compile-run can not be called together with --compile-sbatch\n");
                return 1;
            }
            flags.run_sbatch = true;
            flags.partition = nob_shift(argv, argc);
            if (strcmp(flags.partition, "list") == 0) {
                list_all_partitions();
                return 0;
            } else if (strcmp(flags.partition, "all") != 0 && !partition_is_valid(flags.partition)) {
                list_all_partitions();
                fprintf(stderr,
                        "Error: --run-sbatch received invalid argument '%s', it should be one of: [list,all,partition name...]\n",
                        flags.partition);
                return 1;
            }
        } else if (strcmp(arg, "--out-dir") == 0) {
            if (argc == 0) {
                fprintf(stderr, "Error: --out-dir requires an argument\n");
                return 1;
            }

            flags.out_dir = nob_shift(argv, argc);
        }

        else if (strcmp(arg, "--rebuild") == 0) {
            flags.force_rebuild = true;
        } else if (strcmp(arg, "--ogb") == 0) {
            flags.build_ogb = true;
        } else if (strcmp(arg, "--release") == 0) {
            flags.release = true;
        } else {
            fprintf(stderr, "Got invalid arg: %s\n", arg);
            abort();
        }
    }

    if (flags.build_ogb) {
        char* out_dir = TOOLS_FOLDER;
        char* exec_path = "ogb";
        return build_ogb(exec_path, out_dir);
    }

    else if (flags.compile_sbatch || flags.run_sbatch) {
        if (strcmp(flags.partition, "all") == 0) {
            for (size_t i = 0; i < NOB_ARRAY_LEN(partitions); i++) {
                if (sbatch(partitions[i].name)) return 1;
            }
            return 0;
        }
        else return sbatch(flags.partition);
    }
    else {
        char* out_dir = BUILD_FOLDER;
        char* exec = "paragnn";

        if (flags.out_dir != NULL) {
            out_dir = nob_temp_sprintf("%s%s/", out_dir, flags.out_dir);;
        }
        printf("out_dir: %s\n", out_dir);

        if (!nob_mkdir_if_not_exists(out_dir)) return 1;

        return build_paragnn(exec, out_dir);
    }

    return 0;
}
