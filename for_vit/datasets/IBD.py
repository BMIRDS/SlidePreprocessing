#IBD dataset class
#This script is used with 01_get_svs_meta.py when the IBD dataset is loaded

import json
from pathlib import Path

import pandas as pd

from datasets.base import MetaFile

class IBD(MetaFile):
    def __init__(self, study_name='', svs_path='', json_path='',
                 stratify_by='', num_folds=5):
        super().__init__(study_name, svs_path, json_path, stratify_by,
                         num_folds=num_folds)
        self.df = self.parse_json()
        self.df_svs = self.parse_svs()
        
    # produces self.df by reading in specified fields from input json file
    def parse_json(self):
        fname = self.json_path
        with open(fname) as f:
            d = json.load(f)
        results = []
        for i in range(len(d)):
            if len(d[i]) < 4:
                continue
            results.append([
                d[i]['Slide'],
                d[i]['Class']
            ])

        df = pd.DataFrame(results)
        df.columns = [
            'case_number',
            'Diagnosis'
        ]

        df['study_name'] = self.study_name
        print(df.describe())
        return df
      
    # produces self.df_svs by reading info from svs file names from input svs folder
    def parse_svs(self):
        # creates a list of files found in provided directory
        files = [str(p) for p in Path(self.svs_path).rglob('*.svs')]
        # including 2018 csv data
        files_2018 = [str(p) for p in Path('../../../../datasets/WSI_IBD/svs_2018/').rglob('*.svs')]
        files = files + files_2018
        print(f"{len(files)} files found!")

        df_svs = pd.DataFrame(files, columns=['svs_path'])
        
        # extracting the case numbers from the slides
        case_numbers = []
        for index, row in df_svs.iterrows():
            if row['svs_path'].split('/')[6] == 'svs_2019':
                case_numbers.append(row['svs_path'].split('/')[7])
            else:
                case_numbers.append(row['svs_path'].split('/')[8])
        df_svs['case_number'] = case_numbers
        df_svs['study_name'] = self.study_name

        return df_svs

if __name__ == '__main__':
    # testing to see if id's are extracted succesfully
    files = [str(p) for p in Path('../../../../../datasets/WSI_IBD/svs_2019/').rglob('*.svs')]
    print(f"{len(files)} files found!")
    df_svs = pd.DataFrame(files, columns=['svs_path'])
