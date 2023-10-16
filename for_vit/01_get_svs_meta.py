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

import importlib

from utils.config import Config, default_options
from utils.print_utils import print_intro, print_outro


def main():
    args = default_options()
    config = Config(
        args.default_config_file,
        args.user_config_file)

    module = importlib.import_module(f'datasets.{config.study.dataset_type}')
    Dataset = getattr(module, config.study.dataset_type)
    
    dataset = Dataset(
        study_name=config.study.study_name,
        svs_path=config.study.svs_dir,
        json_path=config.study.json_path,
        stratify_by=config.study.stratify_by,
        num_folds=config.study.num_folds)
    
    dataset.split()
    dataset.make_pickle()


if __name__ == '__main__':
    print_intro(__file__)
    main()
    print_outro(__file__)
