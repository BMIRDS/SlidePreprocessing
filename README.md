# Tools to Preprocess Whole Slides

Author: Naofumi

## Functionalities
### Match slides and corresponding annotation files
[0_match_files.py](0_match_files.py)

### Extract tissues from slides with background masks
[1_extract_tissues.py](1_extract_tissues.py)

### Extract patches from tissues
[2_extract_patches.py](2_extract_patches.py)


## How to run for your dataset
### Config file
In general, you only need to make your own config.yaml file.

There are two cases:
1. svs files and annotations are in the same directory.
2. svs files and annotations are in different directories.

For each case, please see a sample config files:
1. [config_test1.yaml](config_test1.yaml)
2. [config_test2.yaml](config_test2.yaml)

You can run these tests using a [CAMELYON sample](https://www.dropbox.com/s/fegzxxsfycy1shf/testdata.zip?dl=0) (2.7GB). 

For example, using test1 config:
```
$ python 1_extract_tissues.py --config config_test1.yaml
$ python 2_extract_patches.py --config config_test1.yaml
```

### Pairing Slides and Annotations
As the first step of preprocessing, please run 0_match_files.py to make sure the script can locate pairs of slide and xml files.

If it could not locate pairs for all the slides, use `has_xml_user` function in `has_xml.py`. By default, `has_xml` function is used and only works if names of slide and xml files are exactly the same (except its extension). `has_xml_user` function tries advanced formatting to locate files even if xml files have a postfix. It should work well with slides annotated by DHMC pathologists using ASAP software. However, it depends on the naming rules of annotators, so if the function still does not locate corresponding files, try updating the function for your need.

If you use the `has_xml_user` function, please set True for `use_userdefined_has_xml` in config file.


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
