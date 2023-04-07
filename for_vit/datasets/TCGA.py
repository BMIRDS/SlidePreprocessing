#TCGA dataset class
#Example use argument: python 01_get_svs_meta.py --study_name TCGA_COAD --dataset_type TCGA  
# --svs_path /pool2/data/WSI_TCGA/Colorectal  --json_path ./meta-files/TCGA_COAD.json --stratify_by status

import json
from pathlib import Path

import pandas as pd

from datasets.base import MetaFile

class TCGA(MetaFile):
    def __init__(self, study_name='', svs_path='', json_path='',
                 stratify_by='', num_folds=5):
        super().__init__(study_name, svs_path, json_path, stratify_by, num_folds)
        self.df = self.parse_json()
        self.df_svs = self.parse_svs()
    
    #produces self.df by reading in specified fields from input json file
    def parse_json(self):
        
        p = Path(self.json_path)
        with p.open() as f:
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

        print(f"[INFO] Data Frame Shape: {df.shape}")
        print(f"[INFO] Unique Patients: {df.id_patient.unique().shape}")

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
        print("[INFO] ", df.describe())
        df['cancer'] = self.study_name
        return df[[
            'case_id', 'id_patient', 'vital_status', 'days_to_death', 'time',
            'status', 'cancer'
        ]]
    
    #produces self.df_svs by reading info from svs file names from input svs folder
    def parse_svs(self):
        
        files = [str(p) for p in Path(self.svs_path).rglob('*.svs')]
        print(f"[INFO] {len(files)} files found!")

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


