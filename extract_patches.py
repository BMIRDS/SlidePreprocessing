from itertools import product
from multiprocessing.dummy import Pool as ThreadPool
from os import makedirs
from os import walk, makedirs
from os.path import join, basename, dirname, splitext, exists
import argparse

from PIL import Image
from cv2 import imread, imwrite
import openslide
from openslide import OpenSlide
import cv2
import numpy

from extract_tissues import is_purple, is_tissue
from timeit import timeit

IMAGE_EXTS = ['tif', 'svs']
"""In-memory patch extractor, extending extract_tissue script.
Author: Naofumi Tomita
"""


@timeit
def extract_patches_tiff(image_path,
                         dst_dir,
                         opt):
    print("Processing [{}]".format(image_path))
    nonoverlap_factor = 1 - opt.overlap_factor
    try:
        img = OpenSlide(image_path)
    except openslide.lowlevel.OpenSlideUnsupportedFormatError as e:
        print(f"OpenSlideError: {image_path} for {e}")
        return

    compression_factor = opt.compression_factor
    objective_power = img.properties.get('openslide.objective-power')
    if objective_power is None:
        if opt.op_na == 'ignore':
            objective_power = opt.objective_power
        elif opt.op_na == 'skip':
            return
    elif int(objective_power) > opt.objective_power:
        if opt.op_gt == 'ignore':
            objective_power = opt.objective_power
        elif opt.op_gt == 'skip':
            return
        elif opt.op_gt == 'downsample':
            compression_factor *= int(objective_power) / opt.objective_power
    elif int(objective_power) < opt.objective_power:
        if opt.op_lt == 'ignore':
            objective_power = opt.objective_power
        elif opt.op_lt == 'skip':
            return
        elif opt.op_lt == 'upsample':
            compression_factor *= int(objective_power) / opt.objective_power

    x_max, y_max = img.level_dimensions[opt.level]
    ds_factor = img.level_downsamples[opt.level] * compression_factor
    adjusted_window_size = int(opt.size*compression_factor)
    # step size for x and y
    step_size = int(adjusted_window_size * nonoverlap_factor)
    # number of x starting points
    x_steps = int((x_max - adjusted_window_size) / step_size)
    # number of y starting points
    y_steps = int((y_max - adjusted_window_size) / step_size)

    x_max_0, y_max_0 = img.level_dimensions[0]  # (x, y)
    # this operates on the original resolution
    step_size_0 = int(step_size * ds_factor)
    """Extract candidate patches
    """
    print("> Extract candidate patches", flush=True)
    for i in range(x_steps + 1):
        for j in range(y_steps + 1):
            x_start_0 = i * step_size_0
            y_start_0 = j * step_size_0
            top_left_0 = (
                y_start_0, x_start_0
            )  # WARNING: X-Y COORDINATE SHOULD BE SWAPPED. (FOR PIL-OPENCV INCONSISTENCY)
            base = splitext(basename(image_path))[0]
            filename = "_".join([base, str(top_left_0[1]), str(top_left_0[0])])\
                       + ".{}".format(opt.ext)
            out_path = join(dst_dir, filename)
            if exists(out_path):
                continue
            try:
                patch = img.read_region(location=top_left_0,
                                        level=opt.level,
                                        size=(adjusted_window_size,
                                              adjusted_window_size)).convert('RGB')
                if compression_factor != 1:
                    # For compression, extract larger patch and then downsample
                    # to a specified image output size
                    patch = patch.resize(size=(opt.size, opt.size),
                                         resample=Image.BICUBIC)
            except openslide.lowlevel.OpenSlideError as e:
                print(f"OpenSlideError: {image_path} x:{x_start_0} y:{y_start_0} for {e}")
                return

            if is_tissue(patch):
                # if its purple (histopathology images)
                patch.save(out_path)
    print("Finished [{}]".format(image_path), flush=True)


def extract_patches_tiff_batch(opt):
    """
    infolder: .../class/files.jpg
    outfolder:
        if need_class:
            .../dst_dir/class/file/slides.jpg
        else:
            .../dst_dir/file/slides.jpg
    """
    image_paths = list()
    for dirpath, dirs, files in walk(opt.src_dir):
        for file in files:
            if any((ext.lower() in file.split('.')[-1].lower() for ext in IMAGE_EXTS)):
                image_paths.append(join(dirpath, file))

    def process_file(image_path):
        """Make a directory for each WSI
        """
        base = splitext(basename(image_path))[0]
        if opt.keep_class:
            class_name = basename(dirname(image_path))
            dst_dir_split = join(opt.dst_dir, class_name, base)
        else:
            dst_dir_split = join(opt.dst_dir, base)
        makedirs(dst_dir_split, exist_ok=True)
        extract_patches_tiff(image_path,
                             dst_dir_split,
                             opt)

    if opt.num_processes > 0:
        pool = ThreadPool(opt.num_processes)
        coords = pool.starmap(process_file, product(image_paths))
        pool.close()
        pool.join()
    else:
        for image_path in image_paths:
            process_file(image_path)


def load_opt():
    parser = argparse.ArgumentParser(
        description="Extracting patches",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # I/O
    parser.add_argument("--src_dir", '-i', type=str,
                        help="source images")
    parser.add_argument("--dst_dir", '-o', type=str,
                        help="destination of extracted patches")
    # EXTRACTION SETTINGS
    parser.add_argument("--size", '-s', type=int, default=224,
                        help="output patch size")
    parser.add_argument("--overlap_factor", '-f', type=float, default=0,
                        help="overlap ratio of each patches")
    parser.add_argument("--compression_factor", '-c', type=float, default=1.0,
                        help="compression (downscaling) from the selected "
                        "level. use this option if the file level doesn't "
                        "support a desired scale.")
    parser.add_argument("--level", '-l', type=int, default=0,
                        help="target level (see openslide)")
    parser.add_argument("--objective_power", '-j', type=int, default=20,
                        help="base objective power.")
    parser.add_argument("--num_processes", '-p', type=int, default=0,
                        help="multiprocessing. 0 or 1 for sequential. "
                        "Use value > 1 if your CPU has that many threads.")
    # OUTPUT SETTINS
    parser.add_argument("--ext", '-e', type=str, default='png',
                        help="patch file extension")
    parser.add_argument('--keep_class', action='store_true', default=False)
    parser.add_argument('--no_overwrite', action='store_true', default=False)
    # CASE HANGLING
    parser.add_argument('--op_gt', type=str, default='downsample',
                        help="Action when the objective_power of a file is "
                        "greater than the base objective power. "
                        "options: ['ignore', 'downsample', 'skip']")
    parser.add_argument('--op_lt', type=str, default='skip',
                        help="Action when the objective_power of a file is "
                        "less than the base objective power. "
                        "options: ['ignore', 'upsample', 'skip']")
    parser.add_argument('--op_na', type=str, default='skip',
                        help="Action when the objective_power of a file is "
                        "not available or None. options: ['ignore', 'skip']")

    args = parser.parse_args()
    assert args.op_gt in ('ignore', 'downsample', 'skip')
    assert args.op_lt in ('ignore', 'upsample', 'skip')
    assert args.op_na in ('ignore', 'skip')
    return args


if __name__ == '__main__':
    args = load_opt()

    extract_patches_tiff_batch(args)
