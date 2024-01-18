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
import re

import pandas as pd
import tqdm

from utils.config import Config, default_options
from utils.print_utils import print_intro, print_outro
from utils.io_utils import create_patches_meta_path, create_patches_dir, create_slide_meta_path


def main():
    """
    Main function to process and aggregate slide metadata from pickle files.

    This function reads the study configuration, filters metadata based on the study name,
    and aggregates data from individual slide metadata files. It also processes slide IDs,
    patient IDs, and types based on predefined rules and saves the aggregated data to a
    pickle file.
    """
    # Load default options and configuration settings
    args = default_options()
    config = Config(
        args.default_config_file,
        args.user_config_file)

    # Extract the study name from the configuration
    study_name = config.study.study_name

    # Load metadata DataFrame and log its shape
    df_meta = pd.read_pickle(config.patch.svs_meta)
    print("[INFO] ", df_meta.shape)

    # Filter metadata based on the study name
    df_sub = df_meta.loc[df_meta.study_name == study_name]

    # Process each slide ID and aggregate data
    slide_data_frames = []
    for slide_id in tqdm.tqdm(df_sub.id_svs.unique()):
        try:
            # Extract magnification and patch size from config
            magnification = config.patch.magnification
            patch_size = config.patch.patch_size

            # Construct the path for the slide's metadata pickle file
            pickle_file = create_slide_meta_path(study_name, magnification, patch_size, slide_id)
            
            # Load slide metadata and append to the results list
            dfi = pd.read_pickle(pickle_file)
            slide_data_frames.append(dfi)
        except Exception as e:
            print(e)
            print(traceback.format_exc())
    
    # Create a directory for patches based on the study name, magnification, and patch size
    patch_dir = create_patches_dir(study_name, magnification, patch_size)
    #svs_ids = [p.name.replace(config.study.image_extension, '') for p in patch_dir.glob("*")]
    #TODO
    # svs_ids = [p.stem for p in patch_dir.glob("*")] 
    df_sub = pd.read_pickle(config.patch.svs_meta)
    svs_ids = df_sub['id_svs'].tolist()
    
    # Concatenate all slide metadata into a single DataFrame
    df = pd.concat(slide_data_frames)

    # Filter the DataFrame based on slide IDs found in the patches directory
    df = df.loc[df.id_svs.isin(svs_ids)].reset_index(drop=True)
    
    # Extract patient ID and type based on the study naming convention
    if study_name.split('_')[0] == 'TCGA':
        # TODO: NEED TO USE RE
        # Function to extract TCGA patient ID and type using regex
        def extract_tcga_info(path):
            # Regex pattern to match the required TCGA format
            pattern = r'/TCGA-([A-Z0-9-]+)-(\d{2}[A-Z])'
            match = re.search(pattern, path)

            if match:
                # Extracted TCGA patient ID and type
                return match.group(1), match.group(2)
            else:
                # Return None or some default value if pattern not found
                return None, None

        ## OLD CODE BELOW
        # df['id_patient'] = df.file.apply(lambda x: x.split('/')[-3][0:12])
        # df['type'] = df.id_svs.apply(lambda x: x.split('-')[3])
        # TODO: This should work but need to check with real data.
        df['id_patient'], df['type'] = zip(*df['file'].apply(extract_tcga_info))
    else:
        #TODO: This behavior seems dangerous. Shouldn't we have a separated script
        #and describe the input files definition so we don't need to update code
        #based on a dataset?
        df['id_patient'] = df['id_svs']
        df['type'] = '01Z'

    # Log type counts and head of patient IDs and slide IDs for verification
    print("[INFO] ", df.type.value_counts())
    print("[INFO] ", df.id_patient.head())
    print("[INFO] ", df.id_svs.head())

    # Create the path for saving the aggregated metadata
    patch_meta_path = create_patches_meta_path(study_name, magnification, patch_size)
    # Ensure the directory exists before saving
    patch_meta_path.parent.mkdir(parents=True, exist_ok=True)
    # Save the aggregated metadata to a pickle file
    df.to_pickle(patch_meta_path)
    df.head(2).to_csv(str(patch_meta_path).replace('.pickle', '_preview.txt'), index=False, sep='\t')


if __name__ == '__main__':
    print_intro(__file__)
    main()
    print_outro(__file__)
