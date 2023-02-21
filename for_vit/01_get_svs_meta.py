import argparse

from utils.meta import MetaFile, GramStains, TCGA

parser = argparse.ArgumentParser(description='svs/json meta file production')
parser.add_argument('--study_name', type=str, default='gram_stains')
parser.add_argument('--dataset_type', type=str, default='gram_stains')
parser.add_argument('--svs_path', type=str, default='/pool2/users/jackm/dpool/data/svs')
parser.add_argument('--json_path', type=str, default='meta-files/gram_stains.json')
parser.add_argument('--stratify_by', type=str, default='specimen_type')
parser.add_argument('--num_folds', type=str, default= 5)
args = parser.parse_args()

if __name__ == '__main__':
    
    if (args.dataset_type == 'gram_stains'):
        dataset = GramStains(args.study_name, args.svs_path, args.json_path
                                , args.stratify_by, args.num_folds)
    if (args.dataset_type == 'tcga'):
        dataset = TCGA(args.study_name, args.svs_path, args.json_path
                                , args.stratify_by, args.num_folds)
    
    dataset.split()
    dataset.make_pickle()
