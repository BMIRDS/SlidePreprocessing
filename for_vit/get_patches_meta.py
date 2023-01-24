import glob
import pandas as pd
import os
import argparse
import tqdm

parser = argparse.ArgumentParser(description='get patch meta')
parser.add_argument('-c', '--cancer', type=str, default='TCGA_BLCA')
parser.add_argument('-s', '--size', type=int, default=224)
parser.add_argument('-m', '--magnification', type=float, default=10)
parser.add_argument('--svs-meta', type=str, default='meta/meta_svs.pickle')
args = parser.parse_args()

df_meta = pd.read_pickle(args.svs_meta)
print(df_meta.shape)
df_sub = df_meta.loc[df_meta.cancer == args.cancer]
df_sub.id_svs
print(df_sub.shape)
files = []
for subdir in tqdm.tqdm(df_sub.id_svs.unique()):
    files.extend(
        glob.glob(
            f'patches/{args.cancer}/mag_{args.magnification}-size_{args.size}/{subdir}/*/*.jpeg'
        ))
    print(
        f'patches/{args.cancer}/mag_{args.magnification}-size_{args.size}/{subdir}/*/*.jpeg'
    )

df = pd.DataFrame(files, columns=['file'])
df['id_svs'] = df.file.apply(lambda x: x.split('/')[-3])

if args.cancer.split('_')[0] == 'TCGA':
    df['id_patient'] = df.file.apply(lambda x: x.split('/')[-3][0:12])
    df['type'] = df.id_svs.apply(lambda x: x.split('-')[-3])
else:
    df['id_patient'] = df['id_svs']
    df['type'] = '01Z'

print(df.type.value_counts())
print(df.id_patient.head())
print(df.id_svs.head())

os.makedirs(f'meta/{args.cancer}', exist_ok=True)
df.to_pickle(
    f'meta/{args.cancer}/patches_meta-mag_{args.magnification}-size_{args.size}.pickle'
)
