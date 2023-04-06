"""
This script generates metadata files for image datasets by processing .svs files and corresponding .json files. 
It provides a command-line interface to specify the study name, dataset type, file paths, stratification field, 
and the number of folds for k-fold cross-validation.

Usage:

Run the script with appropriate command-line arguments to specify the dataset configuration.
The script imports the specified dataset class from the 'datasets' module.
The dataset class instance is created with the provided configuration, and the dataset is split into training and validation sets.
The resulting metadata is saved as a pickle file.
Command-line arguments:

--study_name: The name of the study to be used (e.g., TCGA_COAD).
--dataset_type: The dataset processing pipeline to use (e.g., TCGA).
--svs_path: The path to the folder containing .svs data (e.g., /pool2/data/WSI_TCGA/Colorectal).
--json_path: The path to the .json file with dataset information (e.g., ./meta-files/TCGA_COAD.json).
--stratify_by: The dataset field to use for stratification (e.g., status).
--num_folds: The number of folds for k-fold cross-validation (e.g., 5).

Example usage:

python script.py --study_name TCGA_COAD --dataset_type TCGA \
                 --svs_path /pool2/data/WSI_TCGA/Colorectal \
                 --json_path ./meta-files/TCGA_COAD.json \
                 --stratify_by status --num_folds 5

"""

import argparse
import importlib

parser = argparse.ArgumentParser(description='svs/json meta file production')
parser.add_argument('--study_name', type=str, required=True,
                    help="Name of the study to be used. "
                    "For example: TCGA_COAD")
parser.add_argument('--dataset_type', type=str, required=True,
                    help="Indicates json processing pipeline. "
                    "For example: TCGA ")
parser.add_argument('--svs_path', type=str, required=True,
                    help="Path to the folder containing svs data. "
                    "For example: /pool2/data/WSI_TCGA/Colorectal ")
parser.add_argument('--json_path', type=str, required=True,
                    help="Path to the json file with dataset info. "
                    "For example: ./meta-files/TCGA_COAD.json ")
parser.add_argument('--stratify_by', type=str, required=True,
                    help="Dataset field to use with stratification. "
                    "For example: status ")
parser.add_argument('--num_folds', type=int, required=True,
                    help= "Number of folds for kfold cross validation. "
                    "For example: 5 ")
args = parser.parse_args()

if __name__ == '__main__':
    
    module = importlib.import_module(f'datasets.{args.dataset_type}')
    Dataset = getattr(module, args.dataset_type)
    
    dataset = Dataset(
        study_name=args.study_name,
        svs_path=args.svs_path,
        json_path=args.json_path,
        stratify_by=args.stratify_by,
        num_folds=args.num_folds)
    
    dataset.split()
    dataset.make_pickle()
