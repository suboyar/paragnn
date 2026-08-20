#ifndef DATASET_INFO_H
#define DATASET_INFO_H

#include <stdbool.h>
#include <stdint.h>

typedef enum { FMT_CSV_GZ, FMT_NPY } RawFormat;

typedef enum {
    DATASET_INVALID,
    DATASET_PROTEINS,
    DATASET_PRODUCTS,
    DATASET_ARXIV,
    DATASET_MAG,
    DATASET_PAPERS100M,
    DATASET_COUNT,
} DatasetKind;

typedef struct {
    const char *name;
    const char *eval_metric;
    const char *task_type;
    const char *download_name;
    const char *version;
    const char *url;
    const char *dir_name; // folder name inside the zip
    const char *split;

    bool add_inverse_edge;
    bool has_node_attr;
    bool has_edge_attr;
    const char *additional_node_files;
    bool is_hetero;
    int64_t feature_count;
    int64_t class_count;
    RawFormat   raw_format;
} DatasetInfo;

extern const DatasetInfo ds_infos[];
DatasetKind str_to_dataset_kind(const char *str);

#endif // DATASET_INFO_H
