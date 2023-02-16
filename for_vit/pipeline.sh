# example pipeline

python get_svs_meta.py
python patch_extraction.py -c=TCGA_COAD --svs-meta=meta/tcga_coad_svs.pickle -m=10 -s=224
python get_patches_meta.py -c=TCGA_COAD --svs-meta=meta/tcga_coad_svs.pickle -m=10 -s=224
python feature_extraction.py -c=TCGA_COAD -m=10 -s=224
python post_process.py -c=TCGA_COAD -m=10 -s=224

