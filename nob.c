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
  nob_cmd_append(&cmd,
                 CC,
                 CFLAGS,
                 "-ggdb",
                 "-fopenmp",
                 "-pg",
                 "-o",
                 BUILD_FOLDER"main",
                 "./main.c",
                 "-lz",);
  if (!nob_cmd_run_sync_and_reset(&cmd)) return 1;

  return 0;
}
