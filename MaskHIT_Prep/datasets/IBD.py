#IBD dataset class
#This script is used with 01_get_svs_meta.py when the IBD dataset is loaded

import json
from pathlib import Path
from scipy.stats import chi2_contingency
import numpy as np
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

    def split(self):
        np.random.seed(45342)  # fixing the seed for reproducibility
        # Extract patient number from case number
        self.df['patient_num'] = self.df.case_number.apply(lambda x: x.split(' ')[0])
            
        # Initialize the 'fold' column
        self.df['fold'] = 0

        unique_patients = self.df['patient_num'].unique()
        np.random.shuffle(unique_patients) 

        # Group data by patient and get summary statistics per patient
        patient_group = self.df.groupby('patient_num').agg({'Diagnosis': pd.Series.mode, 'patient_num': 'count'})

        # patient group consists of most common diagnosis of WSIs for a patient and the total number of slides a patient has
        patient_group = patient_group.rename(columns={'patient_num': 'slide_count'})
        patient_group['Diagnosis'] = patient_group['Diagnosis'].apply(lambda x: x[0] if isinstance(x, np.ndarray) else x)

        # Sort patients by the number of slides (to distribute heavy patients evenly)
        patient_group = patient_group.sort_values(by='slide_count', ascending=False)

        # Distribute patients to folds, aiming to balance both patient count and diagnosis
        fold_diagnosis_counts = {i: {} for i in range(self.num_folds)}
        for patient, data in patient_group.iterrows():
            # Choose the fold with the least representation of this patient's most common diagnosis
            least_rep_fold = min(fold_diagnosis_counts, key=lambda x: fold_diagnosis_counts[x].get(data['Diagnosis'], 0))
            # Assign this patient to that fold
            self.df.loc[self.df['patient_num'] == patient, 'fold'] = least_rep_fold
            # Update fold diagnosis count
            fold_diagnosis_counts[least_rep_fold][data['Diagnosis']] = fold_diagnosis_counts[least_rep_fold].get(data['Diagnosis'], 0) + data['slide_count']
        
        # Display the distribution across folds
        print("[INFO - split] Distribution of data across folds:\n", self.df['fold'].value_counts())
        
        for fold in range(self.num_folds):
            fold_distribution = self.df[self.df['fold'] == fold]['Diagnosis'].value_counts()
            fold_distribution_percent = self.df[self.df['fold'] == fold]['Diagnosis'].value_counts(normalize=True).apply(lambda x: f"{x*100:.2f}%")
            combined_distribution = fold_distribution.astype(str) + " (" + fold_distribution_percent + ")"
            print(f"Fold {fold} diagnosis distribution:\n", combined_distribution)
            
        fold_distributions = [self.df[self.df['fold'] == fold]['Diagnosis'].value_counts() for fold in range(self.num_folds)]
        df_fold_distributions = pd.DataFrame(fold_distributions).fillna(0).T
        print(df_fold_distributions)
        chi2, p, _, _ = chi2_contingency(df_fold_distributions)

        print("Chi-squared test result:", chi2)
        print("p-value:", p)

if __name__ == '__main__':
    # testing to see if id's are extracted succesfully
    files = [str(p) for p in Path('../../../../../datasets/WSI_IBD/svs_2019/').rglob('*.svs')]
    print(f"{len(files)} files found!")
    df_svs = pd.DataFrame(files, columns=['svs_path'])
