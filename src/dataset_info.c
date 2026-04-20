#include <stdbool.h>
#include <string.h>
#include "dataset_info.h"

const DatasetInfo ds_infos[] = {
    [DATASET_ARXIV] = {
        .name         = "arxiv",
        .url          = "https://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip",
        .dir_name     = "arxiv",
        .split_name   = "time",
        .num_nodes    = 169343,
        .num_features = 128,
        .num_classes  = 40,
        .num_edges    = 1166243,
        .undirected   = false,
        .raw_format   = FMT_CSV_GZ,
    },
    [DATASET_PRODUCTS] = {
        .name         = "products",
        .url          = "https://snap.stanford.edu/ogb/data/nodeproppred/products.zip",
        .dir_name     = "products",
        .split_name   = "sales_ranking",
        .num_nodes    = 2449029,
        .num_features = 100,
        .num_classes  = 47,
        .num_edges    = 61859140, // edge count is doubled sisnce inlcudes reverser loops it being undirected
        .undirected   = true,
        .raw_format   = FMT_CSV_GZ,
    },
    [DATASET_PAPERS100M] = {
        .name         = "papers100M",
        .url          = "https://snap.stanford.edu/ogb/data/nodeproppred/papers100M-bin.zip",
        .dir_name     = "papers100M-bin",
        .split_name   = "time",
        .num_nodes    = 111059956,
        .num_features = 128,
        .num_classes  = 172,
        .num_edges    = 1615685872,
        .undirected   = false,
        .raw_format   = FMT_NPY,
    }
};


DatasetKind str_to_dataset_kind(const char *str)
{
    if (strncmp("arxiv", str, 5) == 0)
        return DATASET_ARXIV;
    if (strncmp("products", str, 8) == 0)
        return DATASET_PRODUCTS;
    if (strncmp("papers100M", str, 8) == 0)
        return DATASET_PAPERS100M;

    return DATASET_INVALID;
}
