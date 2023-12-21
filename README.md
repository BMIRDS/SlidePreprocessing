# Slide Preprocessing Tools

This repository contains tools for preparing and processing presentation slides. There are two main tools:

## Slide_Prep (Formerly SlidePreprocessing)
Slide_Prep is designed for processing pathology slides, offering specific functionalities:

- **Matching Slides and Annotations**: Pairs slides with their corresponding annotation files.
- **Tissue Extraction**: Extracts tissues from slides, utilizing background masks.
- **Patch Extraction**: Extracts patches from the identified tissues.

Slide_Prep makes it easier to use these images as datasets for other libraries.

## MaskHIT_Prep (Formerly for_vit)
This tool is for preprocessing slides specifically for the WSI-PLP library. More about WSI-PLP can be found here: [https://github.com/BMIRDS/WSI-PLP](https://github.com/BMIRDS/WSI-PLP).

Both tools help in managing and processing slides. See each tool's directory for instructions and usage.
