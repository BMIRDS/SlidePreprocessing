"""
This script is used for post-processing of extracted image patches to count valid 
patches, and extract and save their corresponding features. The output is saved 
in the data folder as meta data and features.

Usage: The script can be run from the command line with arguments for cancer 
type, magnification, patch size, backbone model, and path to slide meta data. 
The script reads the patch meta data, counts valid patches, merges with patch 
meta data, and extracts and saves corresponding features.

Command-line arguments:

'-c', '--cancer': Type of cancer. Default is 'TCGA_BLCA'.
'-m', '--magnification': Magnification level of the slide. Default is 10.
'-s', '--patch-size': Size of the image patch. Default is 224.
'--backbone': Backbone model for feature extraction. Default is 'resnet_18'.
'--svs-meta': Path to slide meta data. Default is an empty string.

"""

from pathlib import Path
# import argparse
import glob
import itertools
import os

from pandarallel import pandarallel
from skimage.util.shape import view_as_windows
import numpy as np
import pandas as pd
import torch
import tqdm

from utils.config import Config, default_options
from utils.print_utils import print_intro, print_outro
from utils.io_utils import create_patches_meta_path, create_features_dir, create_data_dir

def main():
    args = default_options()
    config = Config(
        args.default_config_file,
        args.user_config_file)

    pandarallel.initialize(nb_workers=config.patch.num_workers, progress_bar=True)

    # parser = argparse.ArgumentParser(description='Post Processing')
    # parser.add_argument('-c', '--cancer', 
    #                     type=str, 
    #                     default='TCGA_BLCA', 
    #                     help='The name of the cancer to process (default: TCGA_BLCA)')
    # parser.add_argument('-m', '--magnification', 
    #                     type=int, 
    #                     default=10, 
    #                     help='The magnification level of the patches (default: 10)')
    # parser.add_argument('-s', '--patch-size', 
    #                     type=int, 
    #                     default=224, 
    #                     help='The size of the patches (default: 224)')
    # parser.add_argument('--backbone', 
    #                     type=str, 
    #                     default='resnet_18', 
    #                     help='The name of the backbone architecture to use (default: resnet_18)')
    # parser.add_argument('--svs-meta', 
    #                     type=str, 
    #                     default='', 
    #                     help='The path to the slide meta file (default: '')')

    # args = parser.parse_args()

    assert Path(config.patch.svs_meta).is_file()

    print("=" * 40)
    print("[INFO] Preparing data")

    study_name = config.study.study_name
    magnification = config.patch.magnification
    patch_size = config.patch.patch_size
    patches_meta_path = create_patches_meta_path(study_name, magnification, patch_size)
    df = pd.read_pickle(patches_meta_path)

    df.groupby('id_svs').sample(1)[[
        'id_patient', 'id_svs', 'type'
    ]].reset_index(drop=True).to_pickle(patches_meta_path.parent / f'svs_meta.pickle')
    # df['folder'] = df.file.apply(lambda x: "/".join(x.split('/')[:4]))

    #TODO: THESE SHOULD BE IMPROVED
    df['pos_x'] = df.file.apply(lambda x: x.split('/')[-2])
    df['pos_y'] = df.file.apply(lambda x: x.split('/')[-1].split('.')[0])
    df['pos_x'] = df['pos_x'].astype(int)
    df['pos_y'] = df['pos_y'].astype(int)


    # need to fix incorrect positions of the patches
    def get_correct_factor(ss):
        aa = ss.unique()
        aa.sort()
        if len(aa) == 1:
            return patch_size
        return np.diff(aa).min()


    df['factor'] = df.groupby('id_svs').pos_x.transform(get_correct_factor)

    df['pos_x'] = df['pos_x'] // df['factor']
    df['pos_y'] = df['pos_y'] // df['factor']
    # df.index = pd.MultiIndex.from_frame(df[['folder','id_patient','id_svs','x','y']])
    # df.drop(columns=['folder','id_patient','id_svs','file','x','y'], inplace=True)

    ########################################
    print("=" * 40)
    print("Counting valid patches")


    def get_counts_chunck(data, delta):
        mask = np.zeros([data.pos_x.max() + delta, data.pos_y.max() + delta])
        mask[data['pos_x'].tolist(), data['pos_y'].tolist()] = 1

        mask_r = view_as_windows(mask, [delta, delta], 1).sum((2, 3))
        _data = pd.DataFrame(list(
            itertools.product(range(mask_r.shape[0]), range(mask_r.shape[1]))),
                            columns=['pos_x', 'pos_y'])
        _data[f'counts_{delta}'] = mask_r[_data['pos_x'], _data['pos_y']]

        _data = _data.merge(data, on=['pos_x', 'pos_y'], how='outer')
        _data.loc[:, f'counts_{delta}'] = _data[f'counts_{delta}'].fillna(0)
        return _data[['pos_x', 'pos_y', f'counts_{delta}']]


    print('\ncount ', 4)
    counts_4 = df[[
        'id_svs', 'pos_x', 'pos_y'
    ]].groupby('id_svs').parallel_apply(lambda x: get_counts_chunck(x, delta=4))
    print('\ncount ', 5)
    counts_5 = df[[
        'id_svs', 'pos_x', 'pos_y'
    ]].groupby('id_svs').parallel_apply(lambda x: get_counts_chunck(x, delta=5))
    print('\ncount ', 10)
    counts_10 = df[[
        'id_svs', 'pos_x', 'pos_y'
    ]].groupby('id_svs').parallel_apply(lambda x: get_counts_chunck(x, delta=10))
    print('\ncount ', 20)
    counts_20 = df[[
        'id_svs', 'pos_x', 'pos_y'
    ]].groupby('id_svs').parallel_apply(lambda x: get_counts_chunck(x, delta=20))
    print('\ncount ', 50)
    counts_50 = df[[
        'id_svs', 'pos_x', 'pos_y'
    ]].groupby('id_svs').parallel_apply(lambda x: get_counts_chunck(x, delta=50))
    print('\ncount ', 100)
    counts_100 = df[[
        'id_svs', 'pos_x', 'pos_y'
    ]].groupby('id_svs').parallel_apply(lambda x: get_counts_chunck(x, delta=100))

    counts_all = counts_4.copy()
    counts_all['counts_5'] = counts_5.counts_5
    counts_all['counts_10'] = counts_10.counts_10
    counts_all['counts_20'] = counts_20.counts_20
    counts_all['counts_50'] = counts_50.counts_50
    counts_all['counts_100'] = counts_100.counts_100
    counts_all = counts_all.reset_index()

    # dfsel = df.loc[df.id_svs == counts_all.loc[counts_all.counts_4 > 16].id_svs.unique()[0]]

    print("")
    print("=" * 40)
    print("[INFO] Merge with patch meta")

    svs_meta = config.patch.svs_meta

    df['valid'] = 1
    df['fid'] = df.index
    df = df.merge(counts_all, on=['id_svs', 'pos_x', 'pos_y'], how='outer')
    df['valid'] = df.valid.fillna(0)

    df_sum = df.groupby('id_svs', as_index=False).valid.sum()
    df_svs = pd.read_pickle(svs_meta)
    if 'valid' in df_svs.columns:
        df_svs.drop('valid', axis=1, inplace=True)

    df_svs = df_svs.merge(df_sum, on='id_svs', how='inner')
    df_svs = df_svs.loc[df_svs.valid > 25]
    df_svs.to_pickle(svs_meta)

    print("")
    print("=" * 40)
    print("[INFO] Combining features")

    num_layers = int(config.patch.backbone.replace('resnet_', ''))
    features_dir = create_features_dir(study_name, magnification, patch_size, num_layers)
    pts = list(features_dir.glob("*.pt"))
    pts.sort()
    features = []
    for pt in tqdm.tqdm(pts):
        features.append(torch.load(pt, map_location='cpu'))
    features = torch.cat(features).detach()

    ##################################################
    # split and save
    ##################################################
    print("=" * 40)
    print("Splitting and save")
    dfs = dict(tuple(df.groupby('id_svs')))

    #TODO: K, V SHOULD BE DESCRIPTIVE: K IS slide_name, 
    for k, v in tqdm.tqdm(dfs.items(), total=len(dfs.keys())):
        dti = features[v.loc[v.valid == 1].fid.astype(int).tolist()]

        data_dir = create_data_dir(study_name, magnification, patch_size, num_layers, k)
        data_dir.mkdir(parents=True, exist_ok=True)

        v['pos'] = v.apply(lambda x: [x['pos_x'], x['pos_y']], axis=1)
        v.reset_index(drop=True)[[
            'pos_x', 'pos_y', 'pos', 'valid', 'counts_4', 'counts_5', 'counts_10',
            'counts_20', 'counts_50', 'counts_100'
        ]].to_pickle(data_dir.parent / "meta.pickle")
        torch.save(dti, data_dir / "features.pt")

if __name__ == '__main__':
    print_intro(__file__)
    main()
    print_outro(__file__)
