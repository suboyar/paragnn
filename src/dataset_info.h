#ifndef DATASET_INFO_H
#define DATASET_INFO_H

#include <stdbool.h>
#include <stdint.h>

typedef enum { FMT_CSV_GZ, FMT_NPY } RawFormat;

typedef enum {
    DATASET_INVALID,
    DATASET_ARXIV,
    DATASET_PRODUCTS,
    DATASET_PAPERS100M,
} DatasetKind;

typedef struct {
    const char *name;
    const char *url;
    const char *dir_name; // folder name inside the zip
    const char *split_name; // name of the folder withing split/ folder
    int64_t     num_nodes;
    int64_t     num_features;
    int64_t     num_classes;
    int64_t     num_edges;
    bool        undirected;
    RawFormat   raw_format;
} DatasetInfo;

extern const DatasetInfo ds_infos[];
DatasetKind str_to_dataset_kind(const char *str);

#endif // DATASET_INFO_H
