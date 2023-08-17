#Gram Stains dataset class
#Example use argument: python 01_get_svs_meta.py --study_name gram_stains --dataset_type GramStains  
# --svs_path /pool2/users/jackm/dpool/data/svs  --json_path meta-files/gram_stains.json --stratify_by specimen_type

import json
from pathlib import Path

import pandas as pd

from datasets.base import MetaFile

class GramStains(MetaFile):
    def __init__(self, study_name='', svs_path='', json_path='',
                 stratify_by='', num_folds=5):
        super().__init__(study_name, svs_path, json_path, stratify_by,
                         num_folds=num_folds)
        self.df = self.parse_json()
        self.df_svs = self.parse_svs()
        
    #produces self.df by reading in specified fields from input json file
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
                d[i]['Tile Annotation'],
                d[i]['Tile Density'],
                d[i]['Focus'],
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
            'image_id',
            'tile_annotation',
            'tile_density',
            'focus',
        ]
    
        # df['id_patient'] = df.id_patient.apply(
        #         lambda x: x.split('-')[1])
        df['id_patient'] = df.image_id
        df['study_name'] = self.study_name
        
        # remove slides that have both gram positive and negativ bacteria
        df.drop(df[df['tile_annotation'] == "Both"].index, inplace = True)
        df.drop(df[df['tile_annotation'] == "None"].index, inplace = True)
        
        print(f"[INFO] Data Frame Shape: {df.shape}")
        print(f"[INFO] Unique Patients: {df.id_patient.unique().shape}")
        print("[INFO] ", df.describe())

        return df
      
    #produces self.df_svs by reading info from svs file names from input svs folder
    def parse_svs(self):
        
        files = [str(p) for p in Path(self.svs_path).rglob('*')]
        print(f"[INFO] {len(files)} files found!")

        df_svs = pd.DataFrame(files, columns=['svs_path'])

        df_svs['id_patient'] = df_svs.svs_path.apply(
            lambda x: Path(x).stem)
        
        df_svs = df_svs.loc[df_svs.id_patient.isin(
            self.df.id_patient)].reset_index(drop=True)

        df_svs['id_svs'] = df_svs.id_patient
        df_svs['study_name'] = self.study_name
        
        self.df = self.df.loc[self.df.id_patient.isin(df_svs.id_patient)].reset_index(drop=True)
        
        return df_svs
    
