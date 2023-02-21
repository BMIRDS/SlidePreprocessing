from pathlib import Path
import glob
import json
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
        
    def split(self):
        
        skf = StratifiedKFold(n_splits=self.num_folds, random_state=45342, shuffle=True)
        print(self.df.size)
        self.df['fold'] = 0
        for i, (train_index,
                test_index) in enumerate(skf.split(self.df, self.df[self.stratify_by])):
            self.df.loc[test_index, 'fold'] = i

        print(pd.crosstab(self.df.fold, self.df[self.stratify_by]))
        
    def make_pickle(self, folder = 'meta'):
        os.makedirs(folder, exist_ok=True)
        self.df.to_pickle(f'{folder}/{self.study_name.lower()}_meta.pickle')
        self.df_svs.to_pickle(f'{folder}/{self.study_name.lower()}_svs.pickle')
        
class GramStains(MetaFile):
    def __init__(self, study_name='', svs_path='', json_path='',
                 stratify_by='', num_folds=5):
        super().__init__(study_name, svs_path, json_path, stratify_by,
                         num_folds=5)
        self.df = self.parse_json()
        self.df_svs = self.parse_svs()
        
    def parse_json(self):
        
        fname = self.json_path
        with open(fname) as f:
            d = json.load(f)
        results = []
        for i in range(len(d)):
            if len(d[i]) < 4:
                continue
            results.append([
                d[i]['Comments (Culture)'],
                d[i]['Gram stain description #1'],
                d[i]['Gram stain description #2'],
                d[i]['Inflam (Quant)'],
                d[i]['Inflammatory cells'],
                d[i]['Morphology'],
                d[i]['Organism (Culture)'],
                d[i]['Quant (1)'],
                d[i]['Quant (2)'],
                d[i]['Quantity (Culture)'],
                d[i]['Specimen type'],
                d[i]['Image ID'],
            ])

        df = pd.DataFrame(results)
        df.columns = [
            'comments_(culture)',
            'gram_stain_description_1',
            'gram_stain_description_2',
            'inflam_(quant)',
            'inflammatory_cells',
            'morphology',
            'organism_(culture)',
            'quant_(1)',
            'quant_(2)',
            'quantity_(culture)',
            'specimen_type',
            'id_patient',
        ]
    
        df['id_patient'] = df.id_patient.apply(
                lambda x: x.split('-')[1])
        df['study_name'] = self.study_name
        
        print(df.shape)
        print(df.id_patient.unique().shape)
        print(df.describe())

        return df
    
    def parse_svs(self):
        
        files = glob.glob(f'{self.svs_path}/*.svs')
        print(f"{len(files)} files found!")

        df_svs = pd.DataFrame(files, columns=['svs_path'])
        df_svs['id_patient'] = df_svs.svs_path.apply(
            lambda x: x.split('-')[1])

        df_svs['id_svs'] = df_svs.svs_path.apply(
            lambda x: x.split('-')[2].replace('.svs', ''))
        df_svs['study_name'] = self.study_name
        
        return df_svs
    
class TCGA(MetaFile):
    def __init__(self, study_name='', svs_path='', json_path='',
                 stratify_by='', num_folds=5):
        super().__init__(study_name, svs_path, json_path, stratify_by,
                         num_folds=5)
        self.df = self.parse_json()
        self.df_svs = self.parse_svs()
        
    def parse_json(self):
        
        fname = self.json_path
        with open(fname) as f:
            d = json.load(f)
        results = []
        for i in range(len(d)):
            if len(d[i]) < 4:
                continue
            results.append([
                d[i]['demographic']['age_at_index'],
                d[i]['demographic']['race'],
                d[i]['demographic']['gender'],
                d[i]['demographic']['vital_status'],
                d[i]['demographic'].get('days_to_death', -1),
                d[i]['demographic']['submitter_id'],
                d[i]['case_id'],
                # d[i]['diagnoses'][0]['tumor_stage'],
                d[i]['diagnoses'][0]['tissue_or_organ_of_origin'],
                d[i]['diagnoses'][0]['days_to_last_follow_up'],
                d[i]['diagnoses'][0].get('ajcc_pathologic_m', 'NA'),
                d[i]['diagnoses'][0].get('ajcc_pathologic_t', 'NA'),
                d[i]['diagnoses'][0].get('ajcc_pathologic_n', 'NA'),
                d[i]['diagnoses'][0].get('primary_diagnosis', '--'),
            ])

        df = pd.DataFrame(results)
        df.columns = [
            'age_at_index',
            'race',
            'gender',
            'vital_status',
            'days_to_death',
            'submitter_id',
            'case_id',
            # 'tumor_stage',
            'tissue_or_organ_of_origin',
            'days_to_last_follow_up',
            'ajcc_pathologic_m',
            'ajcc_pathologic_t',
            'ajcc_pathologic_n',
            'primary_diagnosis'
        ]
        df.submitter_id = df.submitter_id.apply(lambda x: x.split('_')[0])

        df.rename(columns={'submitter_id': 'id_patient'}, inplace=True)

        print(df.shape)
        print(df.id_patient.unique().shape)

        # preparing the survival outcome
        # filtering out patients without follow-up information

        df['time'] = 0
        df.loc[
            df.vital_status == 'Alive',
            'time'] = df[df.vital_status == 'Alive'].days_to_last_follow_up / 365
        df.loc[df.vital_status == 'Dead', 'time'] = [
            np.NaN if x == '--' else int(x) / 365
            for x in df[df.vital_status == 'Dead'].days_to_death.to_list()
        ]
        df['time'] = df.time - df.time.min() + 0.01

        df['status'] = 0
        df.loc[df.vital_status == 'Dead', 'status'] = 1
        print(df.describe())
        df['cancer'] = self.study_name
        return df[[
            'case_id', 'id_patient', 'vital_status', 'days_to_death', 'time',
            'status', 'cancer'
        ]]
    
    def parse_svs(self):
        
        files = glob.glob(f'{self.svs_path}/*/*.svs')
        print(f"{len(files)} files found!")

        df_svs = pd.DataFrame(files, columns=['svs_path'])
        df_svs['id_patient'] = df_svs.svs_path.apply(
            lambda x: x.split('/')[-1][0:12])

        df_svs = df_svs.loc[df_svs.id_patient.isin(
            self.df.id_patient)].reset_index(drop=True)
        df_svs['id_svs'] = df_svs.svs_path.apply(
            lambda x: x.split('/')[-1].replace('.svs', ''))
        df_svs['cancer'] = self.study_name
        df_svs['slide_type'] = df_svs.id_svs.apply(lambda x: x.split('-')[3])

        # only keep the ffpe slides
        df_svs = df_svs.loc[df_svs.slide_type.str[2] == 'Z'].reset_index(drop=True)
        df_svs['slide_type'] = 'ffpe'

        self.df = self.df.loc[self.df.id_patient.isin(df_svs.id_patient)].reset_index(drop=True)
        
        return df_svs

