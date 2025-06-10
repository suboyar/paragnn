#define NOB_EXPERIMENTAL_DELETE_OLD
#define NOB_IMPLEMENTATION
#include "nob.h"

// Some folder paths that we use throughout the build process.
#define CC "gcc"
#define BUILD_FOLDER "./"
#define CFLAGS "-Wall", "-Wextra", "-pedantic", "-O1"

int main(int argc, char **argv)
{
  NOB_GO_REBUILD_URSELF(argc, argv);
  if (!nob_mkdir_if_not_exists(BUILD_FOLDER)) return 1;

  Nob_Cmd cmd = {0};
  // TODO: only compile if src-file is changed
  // TODO: Add -DNDEBUG in release mode to disable assertion
  nob_cmd_append(&cmd,
                 CC,
                 CFLAGS,
                 "-ggdb",
                 "-fopenmp",
                 "-pg",
                 "-o",
                 BUILD_FOLDER"main",
                 "./main.c",
                 "-lm",
                 "-lz",);
  if (!nob_cmd_run_sync_and_reset(&cmd)) return 1;

  return 0;
}


/* TODO: Take inspiration from this on how to use `nob_needs_rebuild`
 * bool build_raylib(void)
 * {
 *     bool result = true;
 *     Nob_Cmd cmd = {0};
 *     Nob_File_Paths object_files = {0};
 *
 *     if (!nob_mkdir_if_not_exists("./build/raylib")) {
 *         nob_return_defer(false);
 *     }
 *
 *     Nob_Procs procs = {0};
 *
 *     const char *build_path = nob_temp_sprintf("./build/raylib/%s", MUSIALIZER_TARGET_NAME);
 *
 *     if (!nob_mkdir_if_not_exists(build_path)) {
 *         nob_return_defer(false);
 *     }
 *
 *     for (size_t i = 0; i < NOB_ARRAY_LEN(raylib_modules); ++i) {
 *         const char *input_path = nob_temp_sprintf(RAYLIB_SRC_FOLDER"%s.c", raylib_modules[i]);
 *         const char *output_path = nob_temp_sprintf("%s/%s.o", build_path, raylib_modules[i]);
 *         output_path = nob_temp_sprintf("%s/%s.o", build_path, raylib_modules[i]);
 *
 *         nob_da_append(&object_files, output_path);
 *
 *         if (nob_needs_rebuild(output_path, &input_path, 1)) {
 *             nob_cmd_append(&cmd, "cc",
 *                 "-ggdb", "-DPLATFORM_DESKTOP", "-D_GLFW_X11", "-fPIC", "-DSUPPORT_FILEFORMAT_FLAC=1",
 *                 "-I"RAYLIB_SRC_FOLDER"external/glfw/include",
 *                 "-c", input_path,
 *                 "-o", output_path);
 *             nob_da_append(&procs, nob_cmd_run_async_and_reset(&cmd));
 *         }
 *     }
 *
 *     if (!nob_procs_wait_and_reset(&procs)) nob_return_defer(false);
 *
 * #ifndef MUSIALIZER_HOTRELOAD
 *     const char *libraylib_path = nob_temp_sprintf("%s/libraylib.a", build_path);
 *
 *     if (nob_needs_rebuild(libraylib_path, object_files.items, object_files.count)) {
 *         nob_cmd_append(&cmd, "ar", "-crs", libraylib_path);
 *         for (size_t i = 0; i < NOB_ARRAY_LEN(raylib_modules); ++i) {
 *             const char *input_path = nob_temp_sprintf("%s/%s.o", build_path, raylib_modules[i]);
 *             nob_cmd_append(&cmd, input_path);
 *         }
 *         if (!nob_cmd_run_sync_and_reset(&cmd)) nob_return_defer(false);
 *     }
 * #else
 *     const char *libraylib_path = nob_temp_sprintf("%s/libraylib.so", build_path);
 *
 *     if (nob_needs_rebuild(libraylib_path, object_files.items, object_files.count)) {
 *         nob_cmd_append(&cmd, "cc", "-shared", "-o", libraylib_path);
 *         for (size_t i = 0; i < NOB_ARRAY_LEN(raylib_modules); ++i) {
 *             const char *input_path = nob_temp_sprintf("%s/%s.o", build_path, raylib_modules[i]);
 *             nob_cmd_append(&cmd, input_path);
 *         }
 *         if (!nob_cmd_run_sync_and_reset(&cmd)) nob_return_defer(false);
 *     }
 * #endif // MUSIALIZER_HOTRELOAD
 *
 * defer:
 *     nob_cmd_free(cmd);
 *     nob_da_free(object_files);
 *     return result;
 * }
 */
