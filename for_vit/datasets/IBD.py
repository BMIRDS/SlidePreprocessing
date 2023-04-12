#Gram Stains dataset class
#Example use argument: python 01_get_svs_meta.py --study_name gram_stains --dataset_type GramStains  
# --svs_path /pool2/users/jackm/dpool/data/svs  --json_path meta-files/gram_stains.json --stratify_by specimen_type

import json
from pathlib import Path

import pandas as pd

from base import MetaFile

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
                d[i]['Case Number '],
                d[i]['Dx (U=UC, C=Cr, I=Ind)'],
                d[i]['Severe AC'],
                d[i]['Mod AC'],
                d[i]['Mild AC'],
                d[i]['Inactive'],
                d[i]['Ileitis'],
                d[i]['Active Ch. Ileitis'],
                d[i]['Dysplasia arising from ACC'],
                d[i]['Granuloma'],
                d[i]['Inflammatory Polyp'],
                d[i]['Architecture Alt.'],
                d[i]['Ileum Inc (y/n)'],
                d[i]['Other'],
                d[i]['1st or Surveillance'],
                d[i]['Slide Comments'],
                d[i]['Unnamed: 16'],
                d[i]['Slide Count'],
                d[i]['Path'],
            ])

        df = pd.DataFrame(results)
        df.columns = [
            'case_number',
            'Dx (U=UC, C=Cr, I=Ind)',
            'Severe AC',
            'Mod AC',
            'Mild AC',
            'Inactive',
            'Ileitis',
            'Active Ch. Ileitis',
            'Dysplasia arising from ACC',
            'Granuloma',
            'Inflammatory Polyp',
            'Architecture Alt.',
            'Ileum Inc (y/n)',
            'Other',
            '1st or Surveillance',
            'Slide Comments',
            'Unnamed: 16',
            'Slide Count',
            'Path'
        ]
    
        df['case_number'] = df.case_number.apply(
                lambda x: x.split('-')[1])
        df['study_name'] = self.study_name
        
        print(df.shape)
        print(df.id_patient.unique().shape)
        print(df.describe())

        return df
      
    # produces self.df_svs by reading info from svs file names from input svs folder
    def parse_svs(self):
        # creates a list of files found in the directory
        files = [str(p) for p in Path(self.svs_path).rglob('*.svs')]
        print(f"{len(files)} files found!")

        df_svs = pd.DataFrame(files, columns=['svs_path'])
        
        # extracting the case numbers from the slides
        df_svs['case_number'] = df_svs.svs_path.apply(
            lambda x: x.split('/')[5])
        df_svs['case_number'] = df_svs.case_number.apply(
            lambda x: x.split(' ')[0])

        # I am not sure how to collect the id_svs information
        # df_svs['id_svs'] = df_svs.svs_path.apply(
        #     lambda x: x.split('-')[2].replace('.svs', ''))

        df_svs['study_name'] = self.study_name
        
        return df_svs

if __name__ == '__main__':
    # testing to see if id's are extracted succesfully
    files = [str(p) for p in Path('/pool2/data/WSI_IBD/svs_2019').rglob('*.svs')]
    print(f"{len(files)} files found!")
    df_svs = pd.DataFrame(files, columns=['svs_path'])

    # extracting the case numbers from the slides
    df_svs['case_number'] = df_svs.svs_path.apply(
        lambda x: x.split('/')[5])
    df_svs['case_number'] = df_svs.case_number.apply(
        lambda x: x.split(' ')[0])

    df_svs.to_csv('with_id_names_extracted.csv')
