"""
This script extracts image patches from whole slide images (WSIs) using the WsiSampler class. 
The extracted patches can be used for training and evaluation of machine learning models in 
various tasks such as cancer diagnosis and tissue classification.

Usage:
python extract_patches.py [-c CANCER] [-j NUM_WORKERS] [-m MAGNIFICATION] [-s PATCH_SIZE] [--svs-meta SVS_META] [--svs-path SVS_PATH]

Command-line arguments:
-c, --cancer CANCER
The cancer subset to process (default: TCGA_BLCA).
-j, --num-workers NUM_WORKERS
Number of parallel workers for processing (default: 10).
-m, --magnification MAGNIFICATION
The magnification level to use for extracting patches (default: 10).
-s, --patch-size PATCH_SIZE
The size of the extracted patches (default: 224).
--svs-meta SVS_META
Path to the pickle file containing metadata for the .svs files (default: meta/dhmc_rcc_svs.pickle).
--svs-path SVS_PATH
Path to a single .svs file for processing. If specified, only this file will be processed.

"""

from pathlib import Path
import traceback

from tqdm.contrib.concurrent import process_map
import pandas as pd

from utils.config import Config, default_options
from utils.print_utils import print_intro, print_outro
from utils.wsi_sampler import WsiSampler
from utils.io_utils import create_slide_meta_dir

def call_get_patches(params):
    return get_patches(*params)

def get_patches(svs_fname: str, svs_root: str, study: str, patch_size: int,
                magnification: float, mag_ori: float, filtering_style: str):
    try:
        wsi = WsiSampler(svs_path=svs_fname,
                         svs_root=svs_root,
                         study=study,
                         mag_mask= 2.5,
                         saturation_enhance=0.5,
                         mag_ori=mag_ori,
                         filtering_style=filtering_style)
        _, save_dirs, _, _, _ = wsi.sample_sequential(0, 100000,
                                                      patch_size,
                                                      magnification)
        svs_fname = Path(svs_fname)
        df = pd.DataFrame(save_dirs, columns=['file'])
        df['id_svs'] = svs_fname.stem
        df['svs_root'] = svs_root

        output_dir = create_slide_meta_dir(study, magnification, patch_size)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{svs_fname.with_suffix('.pickle')}"
        df.to_pickle(output_path)

    except Exception as e:
        # print(inputs)
        print(e)
        print(traceback.format_exc())


def main():
    args = default_options()
    config = Config(
        args.default_config_file,
        args.user_config_file)

    svs_path = config.optional.svs_path
    study_name = config.study.study_name
  
    if svs_path is not None:
        svs_path = Path(svs_path)
        svs_fname = svs_path.name
        svs_folder = str(svs_path.parent)
        paired_inputs = [
            (svs_fname,
             svs_folder, 
             study_name,
             config.patch.patch_size,
             config.patch.magnification,
             config.patch.original_magnification,
             config.patch.filtering_style)]
    else:
        df_sub = pd.read_pickle(config.patch.svs_meta)
        paired_inputs = []
        for i, row in df_sub.iterrows():
            svs_fname = f"{row['id_svs']}{config.study.image_extension}"
            full_path = Path(row['svs_path'])
            svs_folder = str(full_path.parent)
            paired_inputs.append(
                (svs_fname,
                 svs_folder,
                 study_name,
                 config.patch.patch_size,
                 config.patch.magnification,
                 config.patch.original_magnification,
                 config.patch.filtering_style))

    _ = process_map(call_get_patches,
                    paired_inputs, 
                    max_workers=config.patch.num_workers)

if __name__ == '__main__':
    print_intro(__file__)
    main()
    print_outro(__file__)
