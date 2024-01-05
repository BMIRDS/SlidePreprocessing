python 01_get_svs_meta.py --user-config-file configs/config_TCGA_BD.yaml
python 02_patch_extraction.py --user-config-file configs/config_TCGA_BD.yaml
python 03_get_patches_meta.py --user-config-file configs/config_TCGA_BD.yaml
python 04_feature_extraction.py --user-config-file configs/config_TCGA_BD.yaml
python 05_post_process.py --user-config-file configs/config_TCGA_BD.yaml
