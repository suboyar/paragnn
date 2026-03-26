#ifndef DATASET_INFO_H
#define DATASET_INFO_H

typedef enum { FMT_CSV_GZ, FMT_NPY } RawFormat;

typedef enum {
    DATASET_ARXIV,
    DATASET_PRODUCTS,
    DATASET_PAPERS100M,
    DATASET_INVALID,
} DatasetKind;

typedef struct {
    const char *name;
    const char *url;
    uint32_t    num_nodes;
    uint32_t    num_features;
    uint32_t    num_classes;
    uint32_t    num_edges;
    bool        undirected;     // if true num_edges needs to be recomputed at runtime
    RawFormat   raw_format;
} DatasetInfo;

DatasetInfo ds_infos[] = {
    [DATASET_ARXIV] = {
        .name                = "arxiv",
        .url                 = "https://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip",
        .num_nodes           = 169343,
        .num_features        = 128,
        .num_classes         = 40,
        .num_edges           = 1'166'243,
        .undirected          = false,
        .raw_format          = FMT_CSV_GZ,
    },
    [DATASET_PRODUCTS] = {
        .name                = "products",
        .url                 = "https://snap.stanford.edu/ogb/data/nodeproppred/products.zip",
        .num_nodes           = 2'449'029,
        .num_features        = 100,
        .num_classes         = 47,
        .num_edges           = 61'859'140, // edge count is doubled sisnce inlcudes reverser loops it being undirected
        .undirected          = true,
        .raw_format          = FMT_CSV_GZ,
    },
    [DATASET_PAPERS100M] = {
        .name                = "papers100M",
        .url                 = "https://snap.stanford.edu/ogb/data/nodeproppred/papers100M-bin.zip",
        .num_nodes           = 111'059'956,
        .num_features        = 128,
        .num_classes         = 172,
        .num_edges           = 1'615'685'872,
        .undirected          = false,
        .raw_format          = FMT_NPY,
    }
};


static inline DatasetKind str_to_dataset_kind(const char *str)
{
    if (strncmp("arxiv", str, 5) == 0)
        return DATASET_ARXIV;
    if (strncmp("products", str, 8) == 0)
        return DATASET_PRODUCTS;
    if (strncmp("papers100M", str, 8) == 0)
        return DATASET_PAPERS100M;

    return DATASET_INVALID;
}

static inline void list_datasets(void)
{
    printf("Available datasets:\n");
    for (size_t i = 0; i < NOB_ARRAY_LEN(ds_infos); i++)
    {
        printf("  %s\n", ds_infos[i].name);
    }
}

#endif // DATASET_INFO_H
