## MaskHIT_Prep (Formerly for_vit)
This tool is for preprocessing slides specifically for the WSI-PLP library. More about WSI-PLP can be found here: [https://github.com/BMIRDS/WSI-PLP](https://github.com/BMIRDS/WSI-PLP).

## Quick Start Guide
A brief overview for new users handling a new dataset:
1. Create a CSV file describing the dataset with specific column names. See examples in `meta-files/`.
2. Convert CSV to JSON using `scripts/csv_to_json.py`.
3. Write a dataset class under `datasets/` for dataset processing.
4. Create a `user_config` file in `configs/` detailing file paths and metadata. Start by copying `config_default.yaml` and implement based on other config files.
5. Run `pipeline.sh` to execute scripts 01 to 05. In case of issues, run each script manually.

# Installing Dependencies (Recommended method)
For installing necessary dependencies, use the provided script:
`install_requirements.sh`

For Singularity/Docker environment, use the following instead:
`install_requirements_for_container.sh`

## Installation Pitfalls
- `AttributeError: partially initialized module 'cv2' has no attribute 'gapi_wip_gst_GStreamerPipeline'`
    - Cause: Multiple instances of opencv.
    - Fix: `pip uninstall openslide-python`
- `OSError: libopenslide.so.0: cannot open shared object file`
    - Cause: Missing openslide binary.
    - Fix: `apt install openslide-tools python3-openslide`

## Dataset Preparation
### CSV File Structure
Describe the required structure of the CSV file. Include an example.


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

