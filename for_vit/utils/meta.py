# Helper util class for processing meta files
# Designed to be used with 01_get_svs_meta.py
# To add a dataset: Follow format in for_vit/datasets. Extend MetaFile and 
# implement dataset specific parse_json() and parse_svs() functions. These functions 
# are designed to produce self.df and self.df_svs, respectively.

import glob
import os

import pandas as pd
from sklearn.model_selection import StratifiedKFold


class MetaFile:
    def __init__(self, study_name='', svs_path='', json_path='',
                 stratify_by='', num_folds=5):
        self.study_name = study_name
        self.svs_path = svs_path
        self.json_path = json_path
        self.stratify_by = stratify_by
        self.num_folds = num_folds
        
    #Stratified k fold of dataset with a split depending on the given stratify_by argument
    #Split info saved in self.df
    def split(self):
        
        skf = StratifiedKFold(n_splits=self.num_folds, random_state=45342, shuffle=True)
        print(self.df.size)
        self.df['fold'] = 0
        for i, (train_index,
                test_index) in enumerate(skf.split(self.df, self.df[self.stratify_by])):
            self.df.loc[test_index, 'fold'] = i

        print(pd.crosstab(self.df.fold, self.df[self.stratify_by]))
    
    #Produces meta.pickle and svs.pickle files for dataset in specific folder
    def make_pickle(self, folder = 'meta'):
        os.makedirs(folder, exist_ok=True)
        self.df.to_pickle(f'{folder}/{self.study_name.lower()}_meta.pickle')
        self.df_svs.to_pickle(f'{folder}/{self.study_name.lower()}_svs.pickle')
