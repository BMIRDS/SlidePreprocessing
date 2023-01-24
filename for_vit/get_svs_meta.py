import glob
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import os
import json


def parse_json(cancer, root_dir):
    fname = os.path.join(root_dir, f'meta-files/{cancer}.json')
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
    df = df.loc[~df.time.isna()].copy().reset_index(drop=True)
    print('number of participants after excluding missing time %s' %
          df.shape[0])
    print(df.describe())
    return df[[
        'case_id', 'id_patient', 'vital_status', 'days_to_death', 'time',
        'status'
    ]]


if __name__ == '__main__':

    ####################################
    # TCGA_COAD

    # Parameters
    cancer = 'TCGA_COAD'
    site = 'Colorectal'
    svs_root = '/pool2/data'
    stratify_by = 'status'
    num_folds = 5
    ####################################

    df = parse_json(cancer, './')
    df['cancer'] = cancer

    files = glob.glob(f'{svs_root}/WSI_TCGA/{site}/*/*.svs')
    print(f"{len(files)} files found!")

    df_svs = pd.DataFrame(files, columns=['svs_path'])
    df_svs['id_patient'] = df_svs.svs_path.apply(
        lambda x: x.split('/')[-1][0:12])

    df_svs = df_svs.loc[df_svs.id_patient.isin(
        df.id_patient)].reset_index(drop=True)
    df_svs['id_svs'] = df_svs.svs_path.apply(
        lambda x: x.split('/')[-1].replace('.svs', ''))
    df_svs['cancer'] = cancer
    df_svs['slide_type'] = df_svs.id_svs.apply(lambda x: x.split('-')[3])

    # only keep the ffpe slides
    df_svs = df_svs.loc[df_svs.slide_type.str[2] == 'Z'].reset_index(drop=True)
    df_svs['slide_type'] = 'ffpe'

    df = df.loc[df.id_patient.isin(df_svs.id_patient)].reset_index(drop=True)

    skf = StratifiedKFold(n_splits=num_folds, random_state=45342, shuffle=True)
    df['fold'] = 0
    for i, (train_index,
            test_index) in enumerate(skf.split(df, df[stratify_by])):
        df.loc[test_index, 'fold'] = i

    print(pd.crosstab(df.fold, df[stratify_by]))

    os.makedirs('meta', exist_ok=True)
    df.to_pickle(f'meta/{cancer.lower()}_meta.pickle')
    df_svs.to_pickle(f'meta/{cancer.lower()}_svs.pickle')
