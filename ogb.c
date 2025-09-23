/*
  Decodes the ogb-arxiv dataset prepend the relevant metadata and writes it to DATASET_PATH/processed.

  Ussage: gcc -o ogb ./ogb.c -lz && ./ogb
  Optional arguments: -DDATASET_PATH=PATH_TO_OGB_ARXIV
*/

#include <stdio.h>
#include <zlib.h>

#define NOB_IMPLEMENTATION
#include "nob.h"

#include "core.h"


#ifndef DATASET_PATH
    #define DATASET_PATH "./dataset/arxiv"
#endif

Nob_String_Builder sb = {0};

void write_to_entire_file(FILE *f, const char *path, void *data, size_t size)
{
    char *buf = data;
    while (size > 0) {
        size_t n = fwrite(buf, 1, size, f);
        if (ferror(f)) {
            ERROR("Could not write into file %s: %s", path, strerror(errno));
        }
        size -= n;
        buf  += n;
    }
}

#define CHUNK 0x1000 // 4kb window size
void read_gz(Nob_String_Builder *sb, const char* file_path)
{
    gzFile file = gzopen(file_path, "rb");
    if (!file) { ERROR("gzopen of '%s' failed: %s", file_path, strerror(errno)); }

    char buffer[CHUNK];
    while (1) {
        int bytes_read = gzread(file, buffer, CHUNK);
        if (bytes_read <= 0) {
            break;
        }

        nob_sb_append_buf(sb, buffer, bytes_read);
    }

    int err = 0;
    const char *error_string;
    if (!gzeof(file)) {
        error_string = gzerror(file, &err);
    }

    gzclose(file);

    if (err){
        ERROR("gzread of '%s' failed: %s", file_path, error_string);
    }

    nob_sb_append_null(sb);
}

void decode_csv(const char* encode_src, const char* decode_dst)
{
    sb.count = 0;
    read_gz(&sb, encode_src);
    if (sb.items[sb.count-1] == '\0') sb.count--;

    if (!nob_write_entire_file(decode_dst, sb.items, sb.count)) {
        abort();
    }

    printf("Decoded %s -> %s\n", encode_src, decode_dst);
}



int main()
{
    decode_csv(DATASET_PATH"/raw/edge.csv.gz", DATASET_PATH"/processed/edge.csv");
    decode_csv(DATASET_PATH"/raw/node-feat.csv.gz", DATASET_PATH"/processed/node-feat.csv");
    decode_csv(DATASET_PATH"/raw/node-label.csv.gz", DATASET_PATH"/processed/node-label.csv");
    decode_csv(DATASET_PATH"/raw/node_year.csv.gz", DATASET_PATH"/processed/node_year.csv.gz");

    nob_sb_free(sb);
}
