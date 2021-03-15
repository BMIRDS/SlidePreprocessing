# Tools to Preprocess Whole Slides

Author: Naofumi

## Functionalities
### Match slides and corresponding annotation files
[0_match_files.py](0_match_files.py)

### Extract tissues from slides with background masks
[1_extract_tissues.py](1_extract_tissues.py)

### Extract patches from tissues
[2_extract_patches.py](2_extract_patches.py)

### Recent Backward Incompatible Changes
- "flattening" parameter
    - Added in config file
    - Purpose: toggle this to specify the output folder structure. With 'flattening' True, all the extracted tissues are stored in the same folder, which is useful when generating thumbnail images and wanting to see all the images without traversing folders. This also affect folders for patches; all the patches are stored in each folder that corresponds to a class.
    - How to make a previous config file compatible: add "flattening: !!str False" line in config (yaml) file. See [Template file](config/config_template.yaml).


## How to run for your dataset
### Config file
In general, you only need to make your own config.yaml file. [Template file](config/config_template.yaml) is available. This template file also explains each parameter.

There are three use cases:
1. svs files and annotations are in the same directory.
2. svs files and annotations are in different directories.
3. only svs files are available.

For each, please see a sample config files:
1. [config_test1.yaml](config/config_test1.yaml)
2. [config_test2.yaml](config/config_test2.yaml)
3. [config_test3.yaml](config/config_test3.yaml)

You can run these tests using a [CAMELYON sample](https://www.dropbox.com/s/fegzxxsfycy1shf/testdata.zip?dl=0) (2.7GB). 

For example, using test1 config:
```
$ python 1_extract_tissues.py --config config/config_test1.yaml
$ python 2_extract_patches.py --config config/config_test1.yaml
```

For case3, please read [No Annotations section.](#no-annotations)

### Pairing Slides and Annotations
As the first step of preprocessing, please run 0_match_files.py to make sure the script can locate pairs of slide and xml files.

If it could not locate pairs for all the slides, use `has_xml_user` function in `has_xml.py`. By default, `has_xml` function is used and only works if names of slide and xml files are exactly the same (except its extension). `has_xml_user` function tries advanced formatting to locate files even if xml files have a postfix. It should work well with slides annotated by DHMC pathologists using ASAP software. However, it depends on the naming rules of annotators, so if the function still does not locate corresponding files, try updating the function for your need.

If you use the `has_xml_user` function, please set True for `use_userdefined_has_xml` in config file.


### No Annotations?
If slides do not have annotations, and you still want to extract all the crops, please use the following options in config file:

- `action_no_xml` to `process`
- `extract_notannotated` to `True`

If you simply want to skip unannotated slides, set `action_no_xml` to `skip`.

### Thumbnails
It might be helpful for you to see each of whole slides.

To view slides and corresponding annotations,
1. use ASAP to open a slide and load a xml file.
or
2. use `1_extract_tissues.py` script to generate thumbnails. Specifically, set small `target_magnification` and override `thresh_val` parameter in `tissueloc_config` with 1.0 value. Use `config_test1_thumbnail.yaml` as example.
```
$ python 1_extract_tissues.py --config config_test1_thumbnail.yaml
```


## Setup
### Conda environment (optional but recommended)
```
$ conda create --prefix env python=3.7
$ conda activate ./env
```

### Basic Libraries
```
$ pip install numpy
$ pip install pillow
```

### openslide
```
$ conda install -c bioconda openslide-python
$ conda install libiconv
$ pip install scipy
$ pip install scikit-image
$ conda install pixman=0.36.0
```

### opencv (cv2)
```
$ pip install opencv-python
```

### tissueloc (https://github.com/PingjunChen/tissueloc)
```
$ pip install tissueloc
```

### Shapely:
```
$ conda install shapely
```

### YAML:
```
$ pip install pyyaml
```

### Trouble Shooting
- Q: What is "NonAnnotated" folder?
- A: A patch that does not overlap with any ROI annotation is moved to "NonAnnotated" folder. If out-of-ROI regions are normal tissues, then this folder should contain normal tissues, but please make sure if this is true with the original annotators.

- Q: All the patches are classified as "NonAnnotated", although there is a corresponding XML file for a slide. Why?
- A: It happens when the 1) magnification is too low, 2) the ROI is too small and/or 3) stride is too large, thus a patch could not contain enough positive area. The default "tissue_threshold" parameter requires 70% overlap with ROI to consider a patch as a class. You can also lower the parameter at the cost of non-positive tissue noise introduced in a patch.

- Q: There is no patches in "NonAnnotated" folder for a slide. Does this mean all the patches belong to certain classes and no normal tissues?
- A: That's possible. Please double-check if that is true by opening the slide and a corresponding XML file in ASAP or generating a thimbnail image with ROI overlay.


