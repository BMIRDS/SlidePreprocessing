import argparse
import importlib

from utils.meta import MetaFile
from datasets.GramStains import GramStains
from datasets.TCGA import  TCGA

parser = argparse.ArgumentParser(description='svs/json meta file production')
parser.add_argument('--study_name', type=str, help="Name of the study to be used. For example: TCGA_COAD")
parser.add_argument('--dataset_type', type=str, help="Indicates json processing pipeline. For example: TCGA ")
parser.add_argument('--svs_path', type=str, help="Path to the folder containing svs data. For example: /pool2/data/WSI_TCGA/Colorectal ")
parser.add_argument('--json_path', type=str, help="Path to the json file with dataset info. For example: ./meta-files/TCGA_COAD.json ")
parser.add_argument('--stratify_by', type=str, help="Dataset field to use with stratification. For example: status ")
parser.add_argument('--num_folds', type=str, help= "Number of folds for kfold cross validation. For example: 5 ")
args = parser.parse_args()

if __name__ == '__main__':
    
    module = importlib.import_module('datasets.' + args.dataset_type)
    DatasetClass = getattr(module, args.dataset_type)
    
    dataset = DatasetClass(args.study_name, args.svs_path, args.json_path
                                , args.stratify_by, args.num_folds)
    
    dataset.split()
    dataset.make_pickle()
