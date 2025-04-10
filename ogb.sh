#!/usr/bin/env bash

# Take from make_master_file.py in snap-stanford/ogb:
#
# ### add meta-information about protein function prediction task
# name = 'ogbn-proteins'
# dataset_dict[name] = {'num tasks': 112, 'num classes': 2, 'eval metric': 'rocauc', 'task type': 'binary classification'}
# dataset_dict[name]['download_name'] = 'proteins'
# dataset_dict[name]['version'] = 1
# dataset_dict[name]['url'] = 'http://snap.stanford.edu/ogb/data/nodeproppred/'+dataset_dict[name]['download_name']+'.zip'
# ## For undirected grarph, we only store one directional information. This flag allows us to add inverse edge at pre-processing time
# dataset_dict[name]['add_inverse_edge'] = True
# dataset_dict[name]['has_node_attr'] = False
# dataset_dict[name]['has_edge_attr'] = True
# dataset_dict[name]['split'] = 'species'
# dataset_dict[name]['additional node files'] = 'node_species'
# dataset_dict[name]['additional edge files'] = 'None'
# dataset_dict[name]['is hetero'] = False
# dataset_dict[name]['binary'] = False
#
# ### add meta-information about product category prediction task
# name = 'ogbn-products'
# dataset_dict[name] = {'num tasks': 1, 'num classes': 47, 'eval metric': 'acc', 'task type': 'multiclass classification'}
# dataset_dict[name]['download_name'] = 'products'
# dataset_dict[name]['version'] = 1
# dataset_dict[name]['url'] = 'http://snap.stanford.edu/ogb/data/nodeproppred/'+dataset_dict[name]['download_name']+'.zip'
# ## For undirected grarph, we only store one directional information. This flag allows us to add inverse edge at pre-processing time
# dataset_dict[name]['add_inverse_edge'] = True
# dataset_dict[name]['has_node_attr'] = True
# dataset_dict[name]['has_edge_attr'] = False
# dataset_dict[name]['split'] = 'sales_ranking'
# dataset_dict[name]['additional node files'] = 'None'
# dataset_dict[name]['additional edge files'] = 'None'
# dataset_dict[name]['is hetero'] = False
# dataset_dict[name]['binary'] = False
#
# ### add meta-information about arxiv category prediction task
# name = 'ogbn-arxiv'
# dataset_dict[name] = {'num tasks': 1, 'num classes': 40, 'eval metric': 'acc', 'task type': 'multiclass classification'}
# dataset_dict[name]['download_name'] = 'arxiv'
# dataset_dict[name]['version'] = 1
# dataset_dict[name]['url'] = 'http://snap.stanford.edu/ogb/data/nodeproppred/'+dataset_dict[name]['download_name']+'.zip'
# dataset_dict[name]['add_inverse_edge'] = False
# dataset_dict[name]['has_node_attr'] = True
# dataset_dict[name]['has_edge_attr'] = False
# dataset_dict[name]['split'] = 'time'
# dataset_dict[name]['additional node files'] = 'node_year'
# dataset_dict[name]['additional edge files'] = 'None'
# dataset_dict[name]['is hetero'] = False
# dataset_dict[name]['binary'] = False
#
# ### add meta-information about paper venue prediction task
# name = 'ogbn-mag'
# dataset_dict[name] = {'num tasks': 1, 'num classes': 349, 'eval metric': 'acc', 'task type': 'multiclass classification'}
# dataset_dict[name]['download_name'] = 'mag'
# dataset_dict[name]['version'] = 2
# dataset_dict[name]['url'] = 'http://snap.stanford.edu/ogb/data/nodeproppred/'+dataset_dict[name]['download_name']+'.zip'
# dataset_dict[name]['add_inverse_edge'] = False
# dataset_dict[name]['has_node_attr'] = True
# dataset_dict[name]['has_edge_attr'] = False
# dataset_dict[name]['split'] = 'time'
# dataset_dict[name]['additional node files'] = 'node_year'
# dataset_dict[name]['additional edge files'] = 'edge_reltype'
# dataset_dict[name]['is hetero'] = True
# dataset_dict[name]['binary'] = False
#
# ### add meta-information about paper category prediction in huge paper citation network
# name = 'ogbn-papers100M'
# dataset_dict[name] = {'num tasks': 1, 'num classes': 172, 'eval metric': 'acc', 'task type': 'multiclass classification'}
# dataset_dict[name]['download_name'] = 'papers100M-bin'
# dataset_dict[name]['version'] = 1
# dataset_dict[name]['url'] = 'http://snap.stanford.edu/ogb/data/nodeproppred/'+dataset_dict[name]['download_name']+'.zip'
# dataset_dict[name]['add_inverse_edge'] = False
# dataset_dict[name]['has_node_attr'] = True
# dataset_dict[name]['has_edge_attr'] = False
# dataset_dict[name]['split'] = 'time'
# dataset_dict[name]['additional node files'] = 'node_year'
# dataset_dict[name]['additional edge files'] = 'None'
# dataset_dict[name]['is hetero'] = False
# dataset_dict[name]['binary'] = True


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
