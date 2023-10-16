# Helper util class for processing meta files
# Designed to be used with 01_get_svs_meta.py
# To add a dataset: Follow format in for_vit/datasets. Extend MetaFile and 
# implement dataset specific parse_json() and parse_svs() functions. These functions 
# are designed to produce self.df and self.df_svs, respectively.

from pathlib import Path
import abc

import pandas as pd
from sklearn.model_selection import StratifiedKFold


class MetaFile(abc.ABC):
    def __init__(self, study_name: str = '', svs_path: str = '', json_path: str = '',
                 stratify_by: str = '', num_folds: int = 5):
        self.study_name = study_name
        self.svs_path = svs_path
        self.json_path = json_path
        self.stratify_by = stratify_by
        self.num_folds = num_folds
        
    #Stratified k fold of dataset with a split depending on the given stratify_by argument
    #Split info saved in self.df
    def split(self):
        
        skf = StratifiedKFold(n_splits=self.num_folds, random_state=45342, shuffle=True)

        # A new column named "fold" is added to the DataFrame stored in self.df.
        # All rows in this column are initialized to 0.
        self.df['fold'] = 0
        print("[INFO] \n", self.df.to_string())
        splits = skf.split(self.df, self.df[self.stratify_by])
        for i, (_, test_index) in enumerate(splits):
            self.df.loc[test_index, 'fold'] = i

        print(pd.crosstab(self.df.fold, self.df[self.stratify_by]))
    
    #Produces meta.pickle and svs.pickle files for dataset in specific directory
    def make_pickle(self, dir_path: str = 'meta'):
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)

        self.df.to_pickle(dir_path / f'{self.study_name.lower()}_meta.pickle')
        self.df_svs.to_pickle(dir_path / f'{self.study_name.lower()}_svs.pickle')

    @abc.abstractmethod
    def parse_json(self):
        pass

    @abc.abstractmethod
    def parse_svs(self):
        pass
