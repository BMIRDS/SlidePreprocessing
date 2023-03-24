"""
This script processes metadata from a specific cancer dataset, extracting patch 
metadata for a given magnification and patch size. It concatenates the results 
and saves the final metadata as a pickle file.

Usage:
python get_patch_meta.py [-c CANCER] [-s SIZE] [-m MAGNIFICATION] [--svs-meta SVS_META]

Command-line Arguments:
-c, --cancer         : Cancer subset to process (default: TCGA_BLCA)
-s, --size           : Patch size (default: 224)
-m, --magnification  : Magnification level (default: 10)
--svs-meta           : Path to the input metadata pickle file (default: '')

Example:
python get_patch_meta.py -c TCGA_BLCA -s 224 -m 10 --svs-meta meta/dhmc_rcc_svs.pickle

"""

import argparse
import glob
import os

import pandas as pd
import tqdm

parser = argparse.ArgumentParser(description='get patch meta')
parser.add_argument('-c', '--cancer', 
                    type=str, 
                    default='TCGA_BLCA',
                    help='Cancer subset (default: TCGA_BLCA)')
parser.add_argument('-s', '--size', 
                    type=int, 
                    default=224,
                    help='Patch size (default: 224)')
parser.add_argument('-m', '--magnification', 
                    type=int, 
                    default=10,
                    help='Magnification level (default: 10)')
parser.add_argument('--svs-meta', 
                    type=str, 
                    default='',
                    help='Path to the svs meta file (default: None)')
args = parser.parse_args()

df_meta = pd.read_pickle(args.svs_meta)
print(df_meta.shape)
df_sub = df_meta.loc[df_meta.cancer == args.cancer]
res = []
for subdir in tqdm.tqdm(df_sub.id_svs.unique()):
    try:
        dfi = pd.read_pickle(f"meta/{args.cancer}/mag_{args.magnification}-size_{args.size}/{subdir}.pickle")
        res.append(dfi)
    except Exception as e:
        print(e)

svs_ids = [x.split('/')[-1] for x in glob.glob(f"patches/{args.cancer}/mag_{args.magnification}-size_{args.size}/*")]

df = pd.concat(res)
df = df.loc[df.id_svs.isin(svs_ids)].reset_index(drop=True)


if args.cancer.split('_')[0] == 'TCGA':
    df['id_patient'] = df.file.apply(lambda x: x.split('/')[-3][0:12])
    df['type'] = df.id_svs.apply(lambda x: x.split('-')[3])
else:
    df['id_patient'] = df['id_svs']
    df['type'] = '01Z'

print(df.type.value_counts())
print(df.id_patient.head())
print(df.id_svs.head())

os.makedirs(f'meta/{args.cancer}', exist_ok=True)
df.to_pickle(
    f'meta/{args.cancer}/patches_meta-mag_{args.magnification}-size_{args.size}.pickle'
)
