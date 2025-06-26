#include <stdlib.h>

#define NOB_EXPERIMENTAL_DELETE_OLD
#define NOB_IMPLEMENTATION
#include "nob.h"

#define CC "gcc"
#define BUILD_FOLDER "./"
#define EXECUTABLE "./main"
#define WARNING_FLAGS "-Wall", "-Wextra", "-pedantic"
#define RELEASE 0

#define SRC_FILES "./main.c", "./matrix.c"

const char *src_to_obj(const char *src_path);

int main(int argc, char **argv)
{
    int result = EXIT_SUCCESS;
    Nob_Cmd cmd = {0};
    Nob_Procs procs = {0};
    Nob_File_Paths src_files = {0};
    Nob_File_Paths obj_files = {0};

    NOB_GO_REBUILD_URSELF(argc, argv);
    if (!nob_mkdir_if_not_exists(BUILD_FOLDER)) nob_return_defer(EXIT_FAILURE);

    // Skip program name
    nob_shift_args(&argc, &argv);

    bool force_rebuild = false;

    while (argc > 0) {
        const char *arg = nob_shift_args(&argc, &argv);
        if (strcmp(arg, "-B") == 0) {
            force_rebuild = true;
        } else {
            nob_log(NOB_ERROR, "Unknown argument: %s", arg);
            nob_return_defer(EXIT_FAILURE);
        }
    }

    nob_da_append_many(&src_files, ((const char*[]){SRC_FILES}), 2);

    for (size_t i = 0; i < src_files.count; ++i) {
        nob_da_append(&obj_files, src_to_obj(src_files.items[i]));
    }

    // Compile object files
    for (size_t i = 0; i < src_files.count; ++i) {
        const char *src = src_files.items[i];
        const char *obj = obj_files.items[i];

        if (!force_rebuild) {
            int rebuild_needed = nob_needs_rebuild1(obj, src);
            if (rebuild_needed < 0) nob_return_defer(EXIT_FAILURE);
            if (rebuild_needed == 0) continue;
        }

        // Build compile command
        nob_cmd_append(&cmd, CC);
        nob_cmd_append(&cmd, WARNING_FLAGS);
#if RELEASE
        nob_cmd_append(&cmd, "-O3");
        nob_cmd_append(&cmd, "-DNDEBUG");
#else
        nob_cmd_append(&cmd, "-O0");
        nob_cmd_append(&cmd, "-ggdb");
        nob_cmd_append(&cmd, "-pg");
#endif
        nob_cmd_append(&cmd, "-c", src, "-o", obj);

        // Start compilation in background
        Nob_Proc proc = nob_cmd_run_async_and_reset(&cmd);
        if (proc == NOB_INVALID_PROC) nob_return_defer(EXIT_FAILURE);
        nob_da_append(&procs, proc);
    }

    // Wait for all compilations to finish
    if (!nob_procs_wait_and_reset(&procs)) nob_return_defer(EXIT_FAILURE);


    // Link object files to executable
    if (!force_rebuild) {
        int link_needed = nob_needs_rebuild(EXECUTABLE, obj_files.items, obj_files.count);
        if (link_needed < 0) nob_return_defer(EXIT_FAILURE);
        if (link_needed == 0) {
            nob_log(NOB_INFO, "%s is up to date", EXECUTABLE);
            nob_return_defer(EXIT_SUCCESS);
        }
    }
    nob_cmd_append(&cmd, CC);
    // Add object files
    for (size_t i = 0; i < obj_files.count; ++i) {
        nob_cmd_append(&cmd, obj_files.items[i]);
    }
    nob_cmd_append(&cmd,
                   "-fopenmp",
                   "-o", EXECUTABLE,
                   "-lm", "-lz");

    if (!nob_cmd_run_sync_and_reset(&cmd)) return 1;
    nob_log(NOB_INFO, "%s was compiled in %s mode\n",
            EXECUTABLE, RELEASE ? "release" : "debug");

defer:
     nob_cmd_free(cmd);
     nob_da_free(src_files);
     nob_da_free(obj_files);
     nob_da_free(procs);
     return result;
}

const char *src_to_obj(const char *src_path)
{
    Nob_String_Builder sb = {0};
    nob_sb_append_cstr(&sb, src_path);
    // Replace .c with .o
    if (sb.count >= 2 && sb.items[sb.count-2] == '.' && sb.items[sb.count-1] == 'c') {
        sb.items[sb.count-1] = 'o';
    }
    nob_sb_append_null(&sb);
    const char *result = nob_temp_strdup(sb.items);
    nob_sb_free(sb);
    return result;
}
