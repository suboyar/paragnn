/*
 * ParaGNN
 * Copyright (C) 2026  Suleyman Boyar
 *
 * This program is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program.  If not, see <http://www.gnu.org/licenses/>.
 */

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

    DatasetKind datasetkind;
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
