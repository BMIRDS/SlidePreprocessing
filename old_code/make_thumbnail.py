from PIL import Image
from math import ceil
from itertools import product
from multiprocessing.dummy import Pool as ThreadPool

import openslide
from openslide import OpenSlide

from pathlib import Path
from extract_patches import load_opt

"""
Important:
    If you run this script on Ubuntu/Debian, make sure install pixman 0.36.0.
        ref: https://github.com/openslide/openslide-python/issues/62
    conda install pixman=0.36.0
"""
Image.MAX_IMAGE_PIXELS = None

"""
src:
    .../src_dir/file_id/slide.svs

dst:
    .../dst_dir/[<frozen>, <paraffin>]/slide_thumb.jpeg

"""

IMAGE_EXTS = ['.tif', '.svs']


def is_frozen(path: str, frozen_prefix=['BS', 'TS']):
    return any([path.startswith(p) for p in frozen_prefix])


def is_paraffin(path: str, paraffin_prefix=['DX']):
    return any([path.startswith(p) for p in paraffin_prefix])


def make_thumbnail(image_path: Path,
                   dst_dir: Path,
                   opt):
    print("Processing [{}]".format(image_path))
    try:
        img = OpenSlide(str(image_path))
    except openslide.lowlevel.OpenSlideUnsupportedFormatError as e:
        print(f"OpenSlideError: {image_path} for {e}")
        return (image_path.name, f"Failed: {e}")
    compression_factor = opt.compression_factor
    objective_power = img.properties.get('openslide.objective-power')
    if objective_power is None:
        if opt.op_na == 'ignore':
            objective_power = opt.objective_power
        elif opt.op_na == 'skip':
            return (image_path.name, f"Skipped: OP=None")
    elif int(objective_power) > opt.objective_power:
        if opt.op_gt == 'ignore':
            objective_power = opt.objective_power
        elif opt.op_gt == 'skip':
            return (image_path.name, f"Skipped: OP>")
        elif opt.op_gt == 'downsample':
            compression_factor *= int(objective_power) / opt.objective_power
    elif int(objective_power) < opt.objective_power:
        if opt.op_lt == 'ignore':
            objective_power = opt.objective_power
        elif opt.op_lt == 'skip':
            return (image_path.name, f"Skipped: OP<")
        elif opt.op_lt == 'upsample':
            compression_factor *= int(objective_power) / opt.objective_power
    x_max_0, y_max_0 = img.level_dimensions[0]
    x_max, y_max = img.level_dimensions[opt.level]

    ds_factor = img.level_downsamples[opt.level] * compression_factor
    x_max_adjusted = int(x_max//compression_factor)
    y_max_adjusted = int(y_max//compression_factor)
    thumbnail = Image.new(mode='RGB', size=(x_max_adjusted, y_max_adjusted))
    window_size_0 = 10000
    window_size_level = int(window_size_0/img.level_downsamples[opt.level])
    window_size_adjusted = int(window_size_0/ds_factor)
    for c in range(int(ceil(x_max_0/window_size_0))):
        for r in range(int(ceil(y_max_0/window_size_0))):
            try:
                size_x = min(x_max - c*window_size_level, window_size_level)
                size_y = min(y_max - r*window_size_level, window_size_level)
                patch = img.read_region(location=(c*window_size_0, r*window_size_0),
                                        level=opt.level,
                                        size=(size_x, size_y)).convert('RGB')
                if compression_factor != 1:
                    size_x_adjusted = int(size_x/compression_factor)
                    size_y_adjusted = int(size_y/compression_factor)
                    if size_x_adjusted > 0 and size_y_adjusted > 0:
                        patch = patch.resize(size=(size_x_adjusted,
                                                   size_y_adjusted),
                                             resample=Image.BICUBIC)
                thumbnail.paste(patch, box=(c*window_size_adjusted, r*window_size_adjusted))
            except openslide.lowlevel.OpenSlideError as e:
                print(f"OpenSlideError: {image_path} for {e}")
                return (image_path.name, f"Failed: {e}")
    effective_magnification = int(objective_power)/ds_factor
    thumbnail.save(dst_dir/f'{image_path.stem}_@{effective_magnification:.2f}.{opt.ext}')
    return (image_path.name, f"Success")


def make_thumbnails(opt):
    root = Path(opt.src_dir)
    image_paths = [p for p in root.rglob('*') if p.suffix in IMAGE_EXTS and not p.name.startswith('.')]

    def process_file(image_path: Path):
        slide_type = None
        filetype = image_path.name.split('-')[5]
        if is_frozen(filetype):
            slide_type = 'frozen'
        elif is_paraffin(filetype):
            slide_type = 'paraffin'
        else:
            # type unknown
            print("Slide type: unknown")
            return
        dst = Path(opt.dst_dir, slide_type)
        dst.mkdir(parents=True, exist_ok=True)
        result = make_thumbnail(image_path, dst, opt)
        return result

    if opt.num_processes > 0:
        pool = ThreadPool(opt.num_processes)
        results = pool.starmap(process_file, product(image_paths))
        pool.close()
        pool.join()
        print(results)
    else:
        for image_path in image_paths:
            process_file(image_path)


if __name__ == '__main__':
    args = load_opt()
    """ Effective options are:
        src_dir
        dst_dir
        num_processes
        compression_factor
        op_na
        op_gt
        op_lt
        objective_power
        level
        ext
    """
    make_thumbnails(args)
