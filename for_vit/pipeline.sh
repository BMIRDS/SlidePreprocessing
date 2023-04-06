#TODO: OBSOLETE, NEED UPDATE THE PIPELINE. SHOULD USE SMALLER DATASET FOR TESTING.
# example pipeline

python 01_get_svs_meta.py
python 02_patch_extraction.py -c=TCGA_COAD --svs-meta=meta/tcga_coad_svs.pickle -m=10 -s=224
python 03_get_patches_meta.py -c=TCGA_COAD --svs-meta=meta/tcga_coad_svs.pickle -m=10 -s=224
python 04_feature_extraction.py -c=TCGA_COAD -m=10 -s=224
python 05_post_process.py -c=TCGA_COAD -m=10 -s=224

