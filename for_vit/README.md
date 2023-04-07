# Installing Dependencies (Recommended method)
Use `install_requirements.sh`

## Installation Pitfalls
- AttributeError: partially initialized module 'cv2' has no attribute 'gapi_wip_gst_GStreamerPipeline' (most likely due to a circular import)
You have multiple instances of opencv. do pip uninstall openslide-python

- OSError: libopenslide.so.0: cannot open shared object file: No such file or directory
You don't have openslide binary installed. do apt install openslide-tools python3-openslide

# TODO:
## High Priority
- Remove hard-coded data: pool2 paths and cancer types: current default parameters should be noted in the help of each argument.
- Handling data without TCGA-like meta-data. Jack has implemented, and need another validation.
- README should describe the outcome of this scripts. What are we expected to have after running these? This will be specification for the next model pipeline and would give some clues how to process individual dataset.
- Coding standardization: ex) Using pathlib over os.path, os.makedirs, os.glob
- Exception handling should be specific.
- Logging for traceable process
- Documentation: Each script. Each arguments.
- Review default values for arguments. If we set a existing dataset as default, we should provide those metadata as well to serve for pipeline testing.
- Review hardcoded values. ex) PATH_MEAN, PATH_STD
- Unit testing. Should use pytest or similar library for unit test and integration test. 
- README should be more descriptive for both for_vit/ and maskhit/. It should have overview of what the library does, and also a set of tutorials for various input type. For example, see this: https://github.com/facebookresearch/vissl


## Low Priority
- linting (Make sure everyone install/enable pep8 or flake8).


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
