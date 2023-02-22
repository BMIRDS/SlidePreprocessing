import argparse
import importlib

from utils.meta import MetaFile
from datasets.GramStains import GramStains
from datasets.TCGA import  TCGA

parser = argparse.ArgumentParser(description='svs/json meta file production')
parser.add_argument('--study_name', type=str, default='gram_stains')
parser.add_argument('--dataset_type', type=str, default='GramStains')
parser.add_argument('--svs_path', type=str, default='/pool2/users/jackm/dpool/data/svs')
parser.add_argument('--json_path', type=str, default='meta-files/gram_stains.json')
parser.add_argument('--stratify_by', type=str, default='specimen_type')
parser.add_argument('--num_folds', type=str, default= 5)
args = parser.parse_args()

if __name__ == '__main__':
    
    module = importlib.import_module('datasets.' + args.dataset_type)
    DatasetClass = getattr(module, args.dataset_type)
    
    dataset = DatasetClass(args.study_name, args.svs_path, args.json_path
                                , args.stratify_by, args.num_folds)
    
    dataset.split()
    dataset.make_pickle()
