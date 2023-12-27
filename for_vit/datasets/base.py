# Helper util class for processing meta files
# Designed to be used with 01_get_svs_meta.py
# To add a dataset: Follow format in for_vit/datasets. Extend MetaFile and 
# implement dataset specific parse_json() and parse_svs() functions. These functions 
# are designed to produce self.df and self.df_svs, respectively.

from pathlib import Path
from sklearn.model_selection import StratifiedKFold, KFold
import abc
import numpy as np
import pandas as pd


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
        np.random.seed(45342)  # fixing the seed for reproducibility
        if self.study_name == 'IBD_PROJECT':
            # Extract patient number from case number
            self.df['patient_num'] = self.df.case_number.apply(lambda x: x.split(' ')[0])
            
            # Initialize the 'fold' column
            self.df['fold'] = 0

            kf = KFold(n_splits=self.num_folds, random_state=45342, shuffle=True)
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

        else:
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
            # print("[INFO - split] \n", self.df.to_string())

            # Perform the split
            for i, (_, test_index) in enumerate(split_method):
                self.df.loc[test_index, 'fold'] = i

        # Optional: Display the distribution across folds
        print("[INFO - split] Distribution of data across folds:\n", self.df['fold'].value_counts())
        
        for fold in range(self.num_folds):
            print(f"Fold {fold} diagnosis distribution:\n", self.df[self.df['fold'] == fold]['Diagnosis'].value_counts())

    
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
