#include <stdbool.h>
#include <string.h>
#include "dsinfo.h"

const DatasetInfo ds_infos[] = {
    [DATASET_PROTEINS] = {
        .name                  = "ogbn-proteins",
        .eval_metric           = "rocauc",
        .task_type             = "binary classification",
        .download_name         = "proteins",
        .dir_name              = "ogbn_proteins",
        .version               = "1",
        .url                   = "http://snap.stanford.edu/ogb/data/nodeproppred/proteins.zip",
        .add_inverse_edge      = true, // If true, edge_count denotes original directed edges
        .has_node_attr         = false,
        .has_edge_attr         = true,
        .additional_node_files = "node_species",
        .is_hetero             = false,
        .split                 = "species",
        .raw_format            = FMT_CSV_GZ,
        // .node_count            = 132534,
        // .edge_count            = 79122504,
        .feature_count         = 0,
        .class_count           = 2,
    },
    [DATASET_PRODUCTS] = {
        .name                  = "ogbn-products",
        .eval_metric           = "acc",
        .task_type             = "multiclass classification",
        .download_name         = "products",
        .dir_name              = "ogbn_products",
        .version               = "1",
        .url                   = "http://snap.stanford.edu/ogb/data/nodeproppred/products.zip",
        .add_inverse_edge      = true, // If true, edge_count denotes original directed edges
        .has_node_attr         = true,
        .has_edge_attr         = false,
        .additional_node_files = NULL,
        .is_hetero             = false,
        .split                 = "sales_ranking",
        .raw_format            = FMT_CSV_GZ,
        // .node_count            = 2449029,
        // .edge_count            = 123718280,
        .feature_count         = 100,
        .class_count           = 47,
    },
    [DATASET_ARXIV] = {
        .name                  = "ogbn-arxiv",
        .eval_metric           = "acc",
        .task_type             = "multiclass classification",
        .download_name         = "arxiv",
        .dir_name              = "ogbn_arxiv",
        .version               = "1",
        .url                   = "http://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip",
        .add_inverse_edge      = true, // If true, edge_count denotes original directed edges
        .has_node_attr         = true,
        .has_edge_attr         = false,
        .additional_node_files = "node_year",
        .is_hetero             = false,
        .split                 = "time",
        .raw_format            = FMT_CSV_GZ,
        // .node_count            = 169343,
        // .edge_count            = 1166243,
        .feature_count         = 128,
        .class_count           = 40,
    },
    [DATASET_MAG] = {
        .name                  = "ogbn-mag",
        .eval_metric           = "acc",
        .task_type             = "multiclass classification",
        .download_name         = "mag",
        .dir_name              = "ogbn_mag",
        .version               = "2",
        .url                   = "http://snap.stanford.edu/ogb/data/nodeproppred/mag.zip",
        .add_inverse_edge      = false, // If true, edge_count denotes original directed edges
        .has_node_attr         = true,
        .has_edge_attr         = false,
        .additional_node_files = "node_year",
        .is_hetero             = true,
        .split                 = "time",
        .raw_format            = FMT_CSV_GZ,
        // .node_count            = 1939743,
        // .edge_count            = 21111007,
        .feature_count         = 128,
        .class_count           = 349,
    },
    [DATASET_PAPERS100M] = {
        .name                  = "ogbn-papers100M",
        .eval_metric           = "acc",
        .task_type             = "multiclass classification",
        .download_name         = "papers100M-bin",
        .dir_name              = "ogbn_papers100M",
        .version               = "1",
        .url                   = "http://snap.stanford.edu/ogb/data/nodeproppred/papers100M-bin.zip",
        .add_inverse_edge      = true, // If true, edge_count denotes original directed edges
        .has_node_attr         = true,
        .has_edge_attr         = false,
        .additional_node_files = "node_year",
        .is_hetero             = false,
        .split                 = "time",
        .raw_format            = FMT_NPY,
        // .node_count            = 111059956,
        // .edge_count            = 1615685872,
        .feature_count         = 128,
        .class_count           = 172,
    },
};

DatasetKind str_to_dataset_kind(const char *str)
{
    if (strcmp(str, "ogbn-proteins") == 0)
        return DATASET_PROTEINS;
    if (strcmp(str, "ogbn-products") == 0)
        return DATASET_PRODUCTS;
    if (strcmp(str, "ogbn-arxiv") == 0)
        return DATASET_ARXIV;
    if (strcmp(str, "ogbn-mag") == 0)
        return DATASET_MAG;
    if (strcmp(str, "ogbn-papers100M") == 0)
        return DATASET_PAPERS100M;
    return DATASET_INVALID;
}
