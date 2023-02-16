import os
import pandas as pd
from tqdm.contrib.concurrent import process_map
import argparse
from utils.wsi import WsiSampler

parser = argparse.ArgumentParser(description='Patch extraction')
parser.add_argument('-c',
                    '--cancer',
                    type=str,
                    default='TCGA_BLCA',
                    help='cancer subset')
parser.add_argument('-j', '--num-workers', type=int, default=10)
parser.add_argument('-m', '--magnification', type=int, default=10)
parser.add_argument('-s', '--patch-size', type=int, default=224)
parser.add_argument('--svs-meta', type=str, default='meta/dhmc_rcc_svs.pickle')
parser.add_argument('--svs-path',
                    type=str,
                    default=None,
                    help='process single svs')
args = parser.parse_args()




def get_patches(inputs):
    try:
        svs_path, svs_root, study = inputs
        wsi = WsiSampler(svs_path=svs_path,
                         svs_root=svs_root,
                         study=study,
                         saturation_enhance=0.5)
        _, save_dirs, _, _, _ = wsi.sample_sequential(0, 100000,
                                                      args.patch_size,
                                                      args.magnification)
        df = pd.DataFrame(save_dirs, columns=['file'])
        df['id_svs'] = svs_path.replace('.svs', '')
        df['svs_root'] = svs_root

        output_dir = f"meta/{study}/mag_{args.magnification}-size_{args.patch_size}/{svs_path.replace('.svs','.pickle')}"
        os.makedirs(os.path.dirname(output_dir), exist_ok=True)
        df.to_pickle(output_dir)
    except Exception as e:
        print(inputs)
        print(e)



if __name__ == '__main__':
    if args.svs_path is not None:
        svs_fname = os.path.basename(args.svs_path)
        svs_folder = os.path.dirname(args.svs_path)
        paired_inputs = [(svs_fname, svs_folder, args.cancer)]
    else:
        df_sub = pd.read_pickle(args.svs_meta)
        paired_inputs = []
        for i, row in df_sub.iterrows():
            svs_fname = f"{row['id_svs']}.svs"
            full_path = row['svs_path']
            svs_folder = full_path.replace(svs_fname, '')
            paired_inputs.append((svs_fname, svs_folder, args.cancer))

    _ = process_map(get_patches, paired_inputs, max_workers=args.num_workers)
