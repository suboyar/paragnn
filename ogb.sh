#!/usr/bin/env bash

# This metadata is taken from make_master_file.py in the OGB (Open Graph Benchmark) project
# Repository: https://github.com/snap-stanford/ogb
function print_metadata() {
    local dataset="$1"
    
    case "$dataset" in
      proteins)
          echo "name: ogbn-proteins"
          echo "number_of_tasks: 112"
          echo "number_of_classes: 2"
          echo "evaluation_metric: rocauc"
          echo "task_type: binary classification"
          echo "add_inverse_edge: Yes"
          echo "has_node_attributes: No"
          echo "has_edge_attributes: Yes"
          echo "split_type: species"
          echo "additional_node_files: node_species"
          echo "additional_node_files: None"
          echo "is_heterogeneous: No"
          echo "binary_format: No"
          ;;
      products)
          echo "Name: ogbn-products"
          echo "number_of_tasks: 1"
          echo "number_of_classes: 47"
          echo "evaluation_metric: acc"
          echo "task_type: multiclass classification"
          echo "add_inverse_edge: Yes"
          echo "has_node_attributes: Yes"
          echo "has_edge_attributes: No"
          echo "split_type: sales_ranking"
          echo "additional_node_files: None"
          echo "additional_node_files: None"
          echo "is_heterogeneous: No"
          echo "binary_format: No"
          ;;
      arxiv)
          echo "Name: ogbn-arxiv"
          echo "number_of_tasks: 1"
          echo "number_of_classes: 40"
          echo "evaluation_metric: acc"
          echo "task_type: multiclass classification"
          echo "add_inverse_edge: No"
          echo "has_node_attributes: Yes"
          echo "has_edge_attributes: No"
          echo "split_type: time"
          echo "additional_node_files: node_year"
          echo "additional_node_files: None"
          echo "is_heterogeneous: No"
          echo "binary_format: No"
          ;;
      mag)
          echo "Name: ogbn-mag"
          echo "number_of_tasks: 1"
          echo "number_of_classes: 349"
          echo "evaluation_metric: acc"
          echo "task_type: multiclass classification"
          echo "add_inverse_edge: No"
          echo "has_node_attributes: Yes"
          echo "has_edge_attributes: No"
          echo "split_type: time"
          echo "additional_node_files: node_year"
          echo "additional_node_files: edge_reltype"
          echo "is_heterogeneous: Yes"
          echo "binary_format: No"
          ;;
      papers100M-bin)
          echo "Name: ogbn-papers100M"
          echo "number_of_tasks: 1"
          echo "number_of_classes: 172"
          echo "evaluation_metric: acc"
          echo "task_type: multiclass classification"
          echo "add_inverse_edge: No"
          echo "has_node_attributes: Yes"
          echo "has_edge_attributes: No"
          echo "split_type: time"
          echo "additional_node_files: node_year"
          echo "additional_node_files: None"
          echo "is_heterogeneous: No"
          echo "binary_format: yes"
          ;;
      *)
          echo "No metadata available for $dataset"
          ;;
    esac
 }

function downloader() {
    local url="$1"
    local dest_dir="${2:-.}"  # Default to current directory if no argument provided

    # Check if destination exists
    if [ -e "$dest_dir" ]; then
        # Check if it's a directory
        if [ ! -d "$dest_dir" ]; then
            echo "Error: $dest_dir exists but is not a directory"
            return 1
        fi
    else
        # Create directory if it doesn't exist
        mkdir -p "$dest_dir" || {
            echo "Error: Failed to create directory $dest_dir"
            return 1
        }
    fi

    # Download file to specified directory
    wget -P "$dest_dir" "$url"
}

# Parse command line arguments
if [ "$1" == "--metadata" ] && [ -n "$2" ]; then
    # Just print metadata for the specified dataset
    print_metadata "$2"
    exit 0
elif [ "$1" == "--help" ] || [ -z "$1" ]; then
    echo "Usage: $0 [dataset] [destination_directory]"
    echo "       $0 --metadata [dataset]"
    echo ""
    echo "Available datasets: proteins, products, arxiv, mag, papers100M-bin"
    exit 0
else
    case $1 in
      proteins)
          url="http://snap.stanford.edu/ogb/data/nodeproppred/proteins.zip"
          ;;
      products)
          url="http://snap.stanford.edu/ogb/data/nodeproppred/products.zip"
          ;;
      arxiv)
          url="http://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip"
          ;;
      mag)
          url="http://snap.stanford.edu/ogb/data/nodeproppred/mag.zip"
          ;;
      papers100M-bin)
          url="http://snap.stanford.edu/ogb/data/nodeproppred/papers100M-bin.zip"
          ;;
      *)
          echo "Usage: $0 [proteins,products,arxiv,papers100M-bin]"
          exit 1
          ;;
    esac

    downloader $url $2
    print_metadata "$1"
fi
