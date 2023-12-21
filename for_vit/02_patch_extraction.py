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

def get_patches(svs_fname: str, svs_root: str, id_svs: str, study_identifier: str, 
                patch_size: int, magnification: float, mag_original: float, 
                filter_style: str, mag_mask: float):
    try:
        wsi = WsiSampler(svs_path=svs_fname,
                         svs_root=svs_root,
                         id_svs=id_svs,  # Passing id_svs to WsiSampler
                         study_identifier=study_identifier,
                         mag_mask=mag_mask,
                         saturation_enhance=0.5,
                         mag_original=mag_original,
                         filter_style=filter_style)
        _, save_dirs, _, _, _ = wsi.sample_sequential(0, 100000,
                                                      patch_size,
                                                      magnification)
        svs_fname = Path(svs_fname)
        df = pd.DataFrame(save_dirs, columns=['file'])
        df['id_svs'] = id_svs  # Use passed id_svs instead of deriving from svs_fname
        df['svs_root'] = svs_root

        output_dir = create_slide_meta_dir(study_identifier, magnification, patch_size)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{id_svs}.pickle"
        df.to_pickle(output_path)

        # Path for the example text file
        example_file_path = output_dir / f"{id_svs}_example.txt"
        # Save the first two rows of the DataFrame to the text file
        df.head(2).to_csv(example_file_path, index=False, sep='\t')


    except Exception as e:
        # print(inputs)
        print(e)
        print(traceback.format_exc())


def main():
    """
    Main function to process image patches.

    This function initializes the configuration using default and user-provided settings.
    It then determines the source of slide image paths (SVS files) based on the given configuration.

    For each slide image, it either retrieves file details from the configuration or reads
    metadata from a dataframe. It then creates a list of tuples (paired_inputs) containing
    the image filename, folder, study name, and patch configuration parameters.
    These tuples are used as input for the `call_get_patches` function to process each image.
    """

    # Initialize configuration with default and user settings
    args = default_options()
    config = Config(
        args.default_config_file,
        args.user_config_file)

    # Retrieve SVS path and study name from configuration
    svs_path = config.optional.svs_path
    study_name = config.study.study_name
  
    # Process individual SVS file if path is specified
    if svs_path is not None:
        svs_path = Path(svs_path)
        svs_fname = svs_path.name
        svs_folder = str(svs_path.parent)
        id_svs = svs_path.stem  # As no id_svs data give, use stem as ID

        # Create tuples of image processing parameters
        paired_inputs = [
            (svs_fname,
             svs_folder,
             id_svs,
             study_name,
             config.patch.patch_size,
             config.patch.magnification,
             config.patch.original_magnification,
             config.patch.filtering_style,
             config.patch.mag_mask)]
    else:
        # Process a batch of SVS files based on metadata dataframe
        df_sub = pd.read_pickle(config.patch.svs_meta)
        paired_inputs = []
        for i, row in df_sub.iterrows():
            full_path = Path(row['svs_path'])
            svs_fname = full_path.name
            svs_folder = str(full_path.parent)
            id_svs = row['id_svs']

            # Append image processing parameters for each file in the dataframe
            paired_inputs.append(
                (svs_fname,
                 svs_folder,
                 id_svs,
                 study_name,
                 config.patch.patch_size,
                 config.patch.magnification,
                 config.patch.original_magnification,
                 config.patch.filtering_style,
                 config.patch.mag_mask))

    # Call the function to process each image patch
    _ = process_map(call_get_patches,
                    paired_inputs, 
                    max_workers=config.patch.num_workers)

if __name__ == '__main__':
    print_intro(__file__)
    main()
    print_outro(__file__)
