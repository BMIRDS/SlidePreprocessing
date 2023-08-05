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

from pathlib import Path
import traceback

import pandas as pd
import tqdm

from utils.config import Config, default_options
from utils.print_utils import print_intro, print_outro
from utils.io_utils import create_patches_meta_path, create_patches_dir, create_slide_meta_path

def main():
    args = default_options()
    config = Config(
        args.default_config_file,
        args.user_config_file)

    study_name = config.study.study_name
    df_meta = pd.read_pickle(config.patch.svs_meta)
    print("[INFO] ", df_meta.shape)
    df_sub = df_meta.loc[df_meta.study_name == study_name]
    res = []
    for slide_id in tqdm.tqdm(df_sub.id_svs.unique()):
        try:
            
            magnification = config.patch.magnification
            patch_size = config.patch.patch_size
            pickle_file = create_slide_meta_path(study_name, magnification, patch_size, slide_id)
            dfi = pd.read_pickle(pickle_file)
            res.append(dfi)
        except Exception as e:
            print(e)
            print(traceback.format_exc())

    patch_dir = create_patches_dir(study_name, magnification, patch_size)
    #svs_ids = [p.name.replace(config.study.image_extension, '') for p in patch_dir.glob("*")]
    svs_ids = [p.stem for p in patch_dir.glob("*")] 
    
    df = pd.concat(res)
    df = df.loc[df.id_svs.isin(svs_ids)].reset_index(drop=True)
    
    # TODO: NEED TO USE RE
    if study_name.split('_')[0] == 'TCGA':
        df['id_patient'] = df.file.apply(lambda x: x.split('/')[-3][0:12])
        df['type'] = df.id_svs.apply(lambda x: x.split('-')[3])
    else:
        df['id_patient'] = df['id_svs']
        df['type'] = '01Z'
    print("[INFO] ", df.type.value_counts())
    print("[INFO] ", df.id_patient.head())
    print("[INFO] ", df.id_svs.head())

    patch_meta_path = create_patches_meta_path(study_name, magnification, patch_size)
    patch_meta_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(patch_meta_path)


if __name__ == '__main__':
    print_intro(__file__)
    main()
    print_outro(__file__)
