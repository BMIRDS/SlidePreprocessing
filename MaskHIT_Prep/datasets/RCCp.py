#Gram Stains dataset class
#Example use argument: python 01_get_svs_meta.py --study_name gram_stains --dataset_type GramStains  
# --svs_path /pool2/users/jackm/dpool/data/svs  --json_path meta-files/gram_stains.json --stratify_by specimen_type

import json
from pathlib import Path

import pandas as pd

from datasets.base import MetaFile

class RCCp(MetaFile):
    """
    A class for processing RCC profile metadata from JSON and SVS files.

    Extends the MetaFile class to specifically handle RCC profile data. It parses metadata from
    JSON files to create a DataFrame and processes SVS file paths to prepare for patch extraction.

    Attributes:
        df (pandas.DataFrame): DataFrame created from JSON file data.
        df_svs (pandas.DataFrame): DataFrame created from SVS file paths.

    Args:
        study_name (str): The name of the study.
        svs_path (str): The file path to the SVS files.
        json_path (str): The file path to the JSON file containing metadata.
        stratify_by (str): The column name used for stratifying data.
        num_folds (int): The number of folds for stratification.
    """

    def __init__(self, study_name='', svs_path='', json_path='',
                 stratify_by='', num_folds=5):
        super().__init__(study_name, svs_path, json_path, stratify_by,
                         num_folds=num_folds)
        self.df = self.parse_json()
        self.df_svs = self.parse_svs()
        
    #produces self.df by reading in specified fields from input json file
    def parse_json(self):
        """
        Parses the JSON file specified in the class initialization to create a DataFrame.

        This method reads the JSON file, filters and transforms the data based on specific keys,
        and creates a DataFrame with renamed columns for further processing.

        Returns:
            pandas.DataFrame: A DataFrame with selected and renamed fields from the JSON file, or None if an error occurs.
        """

        try:
            with open(self.json_path) as file:
                data = json.load(file)
        except (IOError, json.JSONDecodeError) as e:
            print(f"[ERROR] Failed to read or parse JSON file: {e}")
            return None
        
        # Define required keys to filter the data
        required_keys = {
            'barcode', 'path', 'Sample_Code', 'RCC_Subtype', 
            'Angiogenesis', 'Angio_Score', 'AdenoSig', 'Myeloid_Score'}

        # Extract records with required keys present
        records = [
            {key: record[key] for key in required_keys if key in record}
            for record in data if all(key in record for key in required_keys)
        ]

        # Rename columns for consistency and readability
        df = pd.DataFrame(records)
        df.rename(columns={'Sample_Code': 'sample_code', 
                           'RCC_Subtype': 'rcc_subtype', 
                           'Angiogenesis': 'angiogenesis', 
                           'Angio_Score': 'angio_score', 
                           'AdenoSig': 'adenosig', 
                           'Myeloid_Score': 'myeloid_score'}, inplace=True)
        # Add study name to the DataFrame
        df['study_name'] = self.study_name

        # Logging DataFrame statistics for verification
        print(f"[INFO- parse_json] Data Frame Shape: {df.shape}")
        print(f"[INFO - parse_json] Unique Patients: {df['barcode'].unique().shape}")
        print("[INFO - parse_json] ", df.describe())

        return df

    """
    Note:
    for the next step (02_patch_extraction), this needs to set at least
    - svs_path
    - id_patient
    - id_svs
    """
    #produces self.df_svs by reading info from svs file names from input svs folder
    def parse_svs(self):
        """
        Parses SVS file paths to create a DataFrame with patient and slide information.

        This method checks for necessary columns ('path' and 'barcode') in the existing DataFrame,
        then creates a new DataFrame with SVS paths, extracting patient IDs and slide IDs for further processing.
        It also ensures that only entries with matching barcodes are included in the main DataFrame.

        Returns:
            pandas.DataFrame: A DataFrame containing SVS file paths, patient IDs, and slide IDs, or None if an error occurs.
        """

        # Check for necessary columns in the DataFrame
        if 'path' not in self.df.columns or 'barcode' not in self.df.columns:
            print("[ERROR - parse_svs] Required columns ('path' or 'barcode') not found in DataFrame")
            return None

        # Create DataFrame with SVS file paths
        df_svs = pd.DataFrame(self.df['path'])
        df_svs.columns = ['svs_path'] # Rename the column for clarity


        # Set 'id_patient' to be the same as 'barcode'
        df_svs['id_patient'] = self.df['barcode']

        # Extract 'id_svs' (slide ID) from the SVS file name
        df_svs['id_svs'] = df_svs['svs_path'].apply(lambda x: Path(x).stem.split('.')[0])

        # Add study name to the DataFrame
        df_svs['study_name'] = self.study_name

        # Logging DataFrame statistics for SVS paths
        print("[INFO - parse_svs] ", df_svs.describe())

        # Filter entries in self.df to only include those with barcodes in df_svs
        self.df = self.df[self.df['barcode'].isin(df_svs['id_patient'])].reset_index(drop=True)

        return df_svs
        
