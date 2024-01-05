#!/bin/bash

# Set the configuration path - Replace <<SET_YOUR_OWN_VALUE>> with the actual path to your configuration file
CONFIG_PATH="<SET_YOUR_OWN_VALUE>"

# Check if the configuration file exists
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Configuration file not found at $CONFIG_PATH. Please check the path and try again."
    exit 1
fi

# Running the pipeline with the specified configuration file
echo "Running pipeline with configuration file: $CONFIG_PATH"

python 01_get_svs_meta.py --user-config-file $CONFIG_PATH
python 02_patch_extraction.py --user-config-file $CONFIG_PATH
python 03_get_patches_meta.py --user-config-file $CONFIG_PATH
python 04_feature_extraction.py --user-config-file $CONFIG_PATH
python 05_post_process.py --user-config-file $CONFIG_PATH

echo "Pipeline execution completed."
