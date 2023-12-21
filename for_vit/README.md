## MaskHIT_Prep (Formerly for_vit)
This tool is for preprocessing slides specifically for the WSI-PLP library. More about WSI-PLP can be found here: [https://github.com/BMIRDS/WSI-PLP](https://github.com/BMIRDS/WSI-PLP).

# Installing Dependencies (Recommended method)
For installing necessary dependencies, use the provided script:
`install_requirements.sh`

For Singularity/Docker environment, use the following instead:
`install_requirements_for_container.sh`

## Installation Pitfalls
- `AttributeError: partially initialized module 'cv2' has no attribute 'gapi_wip_gst_GStreamerPipeline' (most likely due to a circular import)`

You may have multiple instances of opencv.

Please do:
`pip uninstall openslide-python`

- `OSError: libopenslide.so.0: cannot open shared object file: No such file or directory`

You don't have openslide binary installed.

Please do:

`apt install openslide-tools python3-openslide`

# For a typical TCGA dataset, please follow `pipeline.sh`

## 1. Process the meta files that include the outcome and the files information
`python 01_get_svs_meta.py`

{cancer}_meta.pickle, one row for each patient, contains the following fields:
* cancer: name of the cancer
* id_patient: patient identifier

{cancer}_svs.pickle, one row for each svs file, contains the following fields:
* cancer
* id_patient
* id_svs: svs file identifier
* slide_type: type of svs file. In TCGA, 01A, 01B etc are frozen slides, 01Z, 02Z etc are FFPE slides

## 2. Extract patches from WSI
`python 02_patch_extraction.py -c=TCGA_COAD --svs-meta=meta/tcga_coad_svs.pickle`
## 3. Obtain a meta file summarising the location of each patch
`python 03_get_patches_meta.py -c=TCGA_COAD --svs-meta=meta/tcga_coad_svs.pickle`
## 4. Extract patches features
`python 04_feature_extraction.py -c=TCGA_COAD -m=10`
## 5. Post process
`python 05_post_process.py -c=TCGA_COAD --svs-meta=meta/tcga_coad_svs.pickle`

