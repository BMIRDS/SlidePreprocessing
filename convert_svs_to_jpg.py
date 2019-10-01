import numpy
from os import walk, makedirs
from os.path import join, basename, dirname
from collections import defaultdict
import pickle
from multiprocessing.dummy import Pool as ThreadPool
from itertools import product
from math import ceil

import scipy.spatial as spatial
import skimage
import cv2
from cv2 import imread, imwrite
from openslide import OpenSlide
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1e10
IMAGE_EXTS = ['JPG', 'JPEG', 'PNG', 'jpg', 'jpeg', 'png']

"""In-memory tissue extractor.
Author: Naofumi Tomita
"""


def is_purple(crop, threshold=100):
    # Credit: Jason Wei
    def is_purple_dot(r, g, b):
        rb_avg = (r+b)/2
        return r > g - 10 and b > g - 10 and rb_avg > g + 20

    pooled = skimage.measure.block_reduce(crop, (int(crop.shape[0]/15), int(crop.shape[1]/15), 1), numpy.average)
    num_purple_squares = 0
    for x in range(pooled.shape[0]):
        for y in range(pooled.shape[1]):
            r = pooled[x, y, 0]
            g = pooled[x, y, 1]
            b = pooled[x, y, 2]
            if is_purple_dot(r, g, b):
                num_purple_squares += 1
    return num_purple_squares > threshold


def is_tissue(img):
    """Checking if it's purple crop.
    """
    color = img.convert('RGB').resize((1, 1), Image.ANTIALIAS).getpixel((0, 0))
    r = color[0] / 255
    g = color[1] / 255
    b = color[2] / 255
    return r > g*1.1 and b > g*1.1 and (r+b)/2 > 0.5


def make_neighborsets(elements, distance=1000):
    """
    elements: a list of tuples, each tuple contains x and y coordinates of point
    """
    points = numpy.array(elements)
    point_tree = spatial.cKDTree(points)
    outer = set([i for i in range(len(elements))])
    inner = set([outer.pop()])
    groups = list()
    while True:
        neighbors = point_tree.query_ball_point(points[list(inner)], distance)
        neighbors = list(set([e for n in neighbors for e in n]))
        old_inner = inner.copy()
        inner.update(neighbors)
        outer.difference_update(inner)
        if inner == set(neighbors) and inner == old_inner:
            groups.append([elements[i] for i in inner])
            if len(outer) == 0:
                break
            inner = set([outer.pop()])
    return groups


def extract_slides_tiff(image_path, out_folder, window_size=112, overlap_factor=1/3, min_patches=3, target_level=0, mask_internal_bg=True):
    """
    Tiff counterpart of extract_slides function
    Extract non-white blocks of tissue from each WSI (image_path)

    min_patches: the minimum number of patches to form a tissue
    mask_internal_bg: True to mask non-purple tissue
    """
    print("Processing [{}]".format(image_path))
    nonoverlap_factor = 1 - overlap_factor

    img = OpenSlide(image_path)

    x_max, y_max = img.level_dimensions[target_level]
    ds_factor = img.level_downsamples[target_level]
    step_size = int(window_size * nonoverlap_factor)  # step size, same for x and y
    x_steps = int((x_max-window_size) / step_size)  # number of x starting points
    y_steps = int((y_max-window_size) / step_size)  # number of y starting points

    x_max_0, y_max_0 = img.level_dimensions[0]  # (x, y)
    step_size_0 = int(step_size*ds_factor)  # this operates on the original resolution

    """Extract candidate patches
    """
    print("> Extract candidate patches")
    def check_candidate(img, x_start, y_start, window_size, target_level):
        top_left = (y_start, x_start)  # WARNING: X-Y COORDINATE SHOULD BE SWAPPED. (FOR PIL-OPENCV INCONSISTENCY)
        patch = img.read_region(
            location=top_left,
            level=target_level,
            size=(window_size, window_size)).convert('RGB')
        if is_tissue(patch) or is_purple(cv2.cvtColor(numpy.array(patch), cv2.COLOR_RGB2BGR)):  # if its purple (histopathology images)
            return (x_start, y_start)

    multiprocess = False
    if multiprocess:
        configs = product(
            [img],
            [i*step_size_0 for i in range(x_steps+1)],
            [j*step_size_0 for j in range(y_steps+1)],
            [window_size],
            [target_level])
        pool = ThreadPool(8*2)
        coords = pool.starmap(check_candidate, configs)
        pool.close()
        pool.join()
        coords = [c for c in coords if c is not None]
    else:
        coords = list()
        for i in range(x_steps+1):
            for j in range(y_steps+1):
                x_start_0 = i*step_size_0
                y_start_0 = j*step_size_0
                top_left_0 = (y_start_0, x_start_0)  # WARNING: X-Y COORDINATE SHOULD BE SWAPPED. (FOR PIL-OPENCV INCONSISTENCY)
                # assert x_start >= 0; assert y_start >= 0; assert x_end <= x_max; assert y_end <= y_max
                patch = img.read_region(
                    location=top_left_0,
                    level=target_level,
                    size=(window_size, window_size)).convert('RGB')
                if is_tissue(patch) or is_purple(cv2.cvtColor(numpy.array(patch), cv2.COLOR_RGB2BGR)):  # if its purple (histopathology images)
                    coords.append(
                        (x_start_0, y_start_0))

    """Form connected components
    """
    print("> Form connected components")
    if len(coords) == 0:
        print("Extraction not applicable to [{}]. Maybe too small?".format(image_path))
        return
    sets = make_neighborsets(coords, distance=window_size*ds_factor)

    """Reconstruct slides
    """
    print("> Reconstruct tissues")
    for each_set in sets:
        if len(each_set) < min_patches:
            continue
        xs = [c[0] for c in each_set]
        ys = [c[1] for c in each_set]
        window_size_0 = int(window_size * ds_factor)
        min_x, max_x = min(xs), max(xs)+window_size_0
        min_y, max_y = min(ys), max(ys)+window_size_0

        lesion_x = int(ceil((max_x-min_x)/ds_factor))
        lesion_y = int(ceil((max_y-min_y)/ds_factor))
        # crop = numpy.ones((lesion_x, lesion_y, 3), numpy.uint8)*255
        crop = numpy.ones((lesion_x, lesion_y, 3), numpy.uint8)*255
        if not mask_internal_bg:
            x_steps = int(ceil((max_x-min_x)/window_size))
            y_steps = int(ceil((max_y-min_y)/window_size))
            each_set = [(min_x+x*window_size, min_y+y*window_size) for x in range(x_steps) for y in range(y_steps)]
        for c in each_set:
            top_left = (c[1], c[0])
            extracted_patch = img.read_region(
                location=top_left,
                level=target_level,
                size=(window_size, window_size)).convert('RGB')
            extracted_patch = cv2.cvtColor(numpy.array(extracted_patch), cv2.COLOR_RGB2BGR)
            # actual_width, actual_height, _ = extracted_patch.shape
            x_start = int((c[0]-min_x)/ds_factor)
            y_start = int((c[1]-min_y)/ds_factor)
            crop[x_start:x_start+window_size,
                 y_start:y_start+window_size, :] = extracted_patch[:min(extracted_patch.shape[0], lesion_x-x_start), :min(extracted_patch.shape[1], lesion_y-y_start), :]
        filename = "_".join([basename(image_path).split('.')[0], str(min_x), str(min_y)])+".jpg"
        out_path = join(out_folder, filename)
        imwrite(out_path, crop)


def extract_slides_tiff_batch(in_folder, out_folder, window_size=112, overlap_factor=1/3, need_class=True, target_level=0, min_patches=5, mask_internal_bg=True):
    """
    infolder: .../class/files.jpg

    outfolder:
        if need_class:
            .../out_folder/class/file/slides.jpg
        else:
            .../out_folder/file/slides.jpg
    """
    image_paths = list()
    for dirpath, dirs, files in walk(in_folder):
        for file in files:
            if any([ext.lower() in file for ext in ['.tif', '.svs']]):
                image_paths.append(join(dirpath, file))
    for image_path in image_paths:
        """Make a directory for each WSI
        """
        base = basename(image_path).split('.')[0]
        if need_class:
            class_name = basename(dirname(image_path))
            out_folder_split = join(out_folder, class_name, base)
        else:
            out_folder_split = join(out_folder, base)
        makedirs(out_folder_split, exist_ok=True)
        extract_slides_tiff(
            image_path,
            out_folder_split,
            window_size=window_size,
            overlap_factor=overlap_factor,
            target_level=target_level,
            min_patches=min_patches,
            mask_internal_bg=mask_internal_bg)


if __name__ == '__main__':
    "Function call example"
    in_folder = 'svs_folder'
    out_folder = 'output_folder'
    extract_slides_tiff_batch(in_folder, out_folder, need_class=False, overlap_factor=1/3, target_level=0, mask_internal_bg=False)
