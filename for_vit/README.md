# For a typical TCGA dataset, please follow `pipeline.sh`

# Step 1. Process the meta files that include the outcome and the files information
`python get_svs_meta.py`

{cancer}_meta.pickle, one row for each patient, contains the following fields:
* cancer: name of the cancer
* id_patient: patient identifier

{cancer}_svs.pickle, one row for each svs file, contains the following fields:
* cancer
* id_patient
* id_svs: svs file identifier
* slide_type: type of svs file. In TCGA, 01A, 01B etc are frozen slides, 01Z, 02Z etc are FFPE slides

# Step 2: Extract patches from WSI
`python patch_extraction.py -c=TCGA_COAD --svs-meta=meta/tcga_coad_svs.pickle`
# Step 3: Obtain a meta file summarising the location of each patch
`python get_patches_meta.py -c=TCGA_COAD --svs-meta=meta/tcga_coad_svs.pickle`
# Step 4: Extract patches features
`python feature_extraction.py -c=TCGA_COAD -m=10`
# Step 5: Post process
`python post_process.py -c=TCGA_COAD`




