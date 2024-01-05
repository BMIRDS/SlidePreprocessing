# Helper util class for processing meta files
# Designed to be used with 01_get_svs_meta.py
# To add a dataset: Follow format in for_vit/datasets. Extend MetaFile and 
# implement dataset specific parse_json() and parse_svs() functions. These functions 
# are designed to produce self.df and self.df_svs, respectively.

from pathlib import Path
import abc

import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold


class MetaFile(abc.ABC):
    def __init__(self, study_name: str = '', svs_path: str = '', json_path: str = '',
                 stratify_by: str = '', num_folds: int = 5):
        self.study_name = study_name
        self.svs_path = svs_path
        self.json_path = json_path
        self.stratify_by = stratify_by
        self.num_folds = num_folds
    
    # Split dataset into k folds, with optional stratification
    def split(self):
        if self.stratify_by is not None and self.stratify_by in self.df.columns:
            # Use StratifiedKFold if stratification is required
            kf = StratifiedKFold(n_splits=self.num_folds, random_state=45342, shuffle=True)
            split_method = kf.split(self.df, self.df[self.stratify_by])
        else:
            # Use KFold if no stratification is required
            kf = KFold(n_splits=self.num_folds, random_state=45342, shuffle=True)
            split_method = kf.split(self.df)

        # Initialize the "fold" column to 0
        self.df['fold'] = 0
        print("[INFO - split] \n", self.df.to_string())

        # Perform the split
        for i, (_, test_index) in enumerate(split_method):
            self.df.loc[test_index, 'fold'] = i

        # Optional: Display the distribution across folds
        print("[INFO - split] Distribution of data across folds:\n", self.df['fold'].value_counts())

    
    #Produces meta.pickle and svs.pickle files for dataset in specific directory
    def make_pickle(self, dir_path: str = 'meta'):
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)

        df_meta_path = dir_path / f'{self.study_name.lower()}_meta.pickle'
        self.df.to_pickle(df_meta_path)
        df_svs_path = dir_path / f'{self.study_name.lower()}_svs.pickle'
        self.df_svs.to_pickle(df_svs_path)

        # Example (Preview) files
        self.df.head(2).to_csv(
            str(df_meta_path).replace(".pickle", "_preview.txt"), 
            index=False, sep='\t')
        self.df_svs.head(2).to_csv(
            str(df_svs_path).replace(".pickle", "_preview.txt"), 
            index=False, sep='\t')

        # Remind to set the object paths to the user.
        print(f"[INFO - make_pickle] Make sure to set the svs meta file path in your config file.")
        print(f"  svs_meta: !!str {df_svs_path}")

    @abc.abstractmethod
    def parse_json(self):
        pass

    @abc.abstractmethod
    def parse_svs(self):
        pass
