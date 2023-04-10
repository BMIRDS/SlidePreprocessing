"""
This preprocessing script is used to divide an svs slide into smaller tiles for annotation. This is helpful when one
slide may contain sparse organisms/different identities in different areas of the whole slide image. 
It's used for the Gram Stains dataset to improve annotation by allowing identification of which slide areas
contain bacteria. Users can specify the tile size and compression. The resulting tiles can be manually annotated
using the annotation folder with entries input by the user into the "Tile Annotation" column of the output cvs.
The output cvs and the tile folder of .tif images are then used as inputs to 01_get_svs_meta.py
From there preprocessing and training proceed as usual. Make sure that your datset in 01_get_svs_meta.py is
set up to handle tile inputs and that it looks for the right column in the csv

To run, provide a folder of svs slides along with a csv file containing info for each slide.
Each svs slide must have a corresponding entry in the csv file under the column "Image ID".
ie. the svs file 21s-007mi0082-2.svs would have a csv entry 21s-007mi0082-2 under "Image ID"

Config arguments:
--svs_dir: path to svs files
--tile_folder: where to save produced tif tiles
--annotation_folder: where to save produced jpeg files for annotation
--csv_in_path: input csv with svs information
--csv_out_path: where to save csv with tile entries
--compression_factor: How much to compress jpeg and tiff
--window_size: Window size for tile production
                       
Adaptation of tile indexing from:
https://github.com/BMIRDS/deepslide/blob/master/code/z_preprocessing/2_svs_to_jpg_tiles.py
"""

import argparse
from pathlib import Path
from math import ceil

import openslide
from PIL import Image
import pyvips
import pandas as pd

from utils.config import Config, default_options
from utils.print_utils import print_intro, print_outro

def output_jpeg_tiles(image_name, anno_subdir, tile_path, compression_factor, window_size):  
  
  img = openslide.OpenSlide(image_name)
  width, height = img.level_dimensions[0]
    
  increment_x = int(ceil(width / window_size))
  increment_y = int(ceil(height / window_size))

  print("converting", image_name, "with width", width, "and height", height)
    
  index = 1

  for incre_x in range(increment_x):
    for incre_y in range(increment_y):

      begin_x = window_size * incre_x
      end_x = min(width, begin_x + window_size)
      begin_y = window_size * incre_y
      end_y = min(height, begin_y + window_size)
      tile_width = end_x - begin_x
      tile_height = end_y - begin_y

      tile = img.read_region((begin_x, begin_y), 0, (tile_width, tile_height))
      tile.load()
      tile_rgb = Image.new("RGB", tile.size, (255, 255, 255))
      tile_rgb.paste(tile, mask=tile.split()[3])
 

      # compress the image
      tile_rgb = tile_rgb.resize((int(tile_rgb.size[0] / compression_factor), int(tile_rgb.size[1] / compression_factor)), Image.ANTIALIAS)

      # save the image
      output_image_name = anno_subdir + '/' + image_name.split('/')[-1][:-4] + '_' + f"{index:03d}" + '.jpg'
      tile_rgb.save(output_image_name)
            
      #convert to tiff
      output_tif_name = tile_path + '_' + f"{index:03d}" + '.tif'
      vipsimage = pyvips.Image.new_from_file(output_image_name)
      vipsimage.tiffsave(output_tif_name, compression="jpeg", tile=True, tile_width=512, tile_height=512, pyramid=True, bigtiff=True)
      
      index += 1
    
  return index      

def main():
  args = default_options()
  config = Config(
      args.default_config_file,
      args.user_config_file)
    
  src = Path(config.study.svs_dir)
  tile = Path(config.tiles.tile_dir)
  anno = Path(config.tiles.annotation_dir)
  tile.mkdir(exist_ok=True)
  anno.mkdir(exist_ok=True)
    
  files = [str(p) for p in src.rglob('*.svs')]
  
  in_csv = pd.read_csv(config.tiles.csv_in_path)
  in_csv.set_index("Image ID", inplace=True)
  # "Tile Annotation" column set blank, requires manual labels after tile processing
  in_csv.insert(0, 'Tile Annotation', '')
  in_csv.insert(0, 'Tile ID', '')
  tile_csv = pd.DataFrame()

  for f in files:
    name = f.split('/')[-1]
    anno_path = config.tiles.annotation_dir + '/' + name[:-4]
    subfolder_dir = Path(anno_path)
    subfolder_dir.mkdir(exist_ok=True)
    tile_path = config.tiles.tile_dir + '/' + name.split('.')[0]

    count = output_jpeg_tiles(f, anno_path, tile_path, config.tiles.tile_compression_factor, config.tiles.tile_window_size)
    
    # update csv with new rows for each tile
    image_id = name.split('.')[0]
    
    row = in_csv.loc[image_id].copy()
    for i in range (1, count + 1):    
      new_row = row
      # Identification by Tile ID column set for future processing, replaces "Image ID" and matches name of tif file
      new_row.loc["Tile ID"] = image_id + '_' + f"{i:03d}"
      tile_csv = tile_csv.append(new_row)
      
  tile_csv.to_csv(config.tiles.csv_out_path, index=False)
  
if __name__ == '__main__':
    print_intro(__file__)
    main()
    print_outro(__file__)
    
    
