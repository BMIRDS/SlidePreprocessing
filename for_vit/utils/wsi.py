import os
from pathlib import Path

import numpy as np
import PIL
import PIL.ImageEnhance
from PIL import Image
from skimage.transform import resize
import openslide

def get_original_magnification(slide: openslide.OpenSlide):
    # Adapted from: https://github.com/BMIRDS/TCGA-SlidePipeline/blob/main/SlidePrep/slideprep/utils/microscope.py
    ERR_RANGE = 10  # in %
    ERR_RANGE = ERR_RANGE / 100
    objective_power = None
    if 'openslide.objective-power' in slide.properties:
        objective_power = float(slide.properties.get('openslide.objective-power'))

    elif 'openslide.mpp-x' in slide.properties:
        objective_power = 10 / float(slide.properties.get('openslide.mpp-x'))

    else:
        print("[INFO] Magnification is not available.")
        return objective_power

    """Objective Power is often not accurate; it should typically be 10, 20, or
    40. Thus need rounding. ex) 41.136... -> 40
    """
    objective_power_candidates = [10, 20, 40, 60]
    calibrated = False
    for candidate in objective_power_candidates:
        if candidate * (1 - ERR_RANGE) <= objective_power and\
         objective_power <= candidate * (1 + ERR_RANGE):
            objective_power = candidate
            calibrated = True
            break
    if not calibrated:
        print(f"[INFO] Magnification is not standard value: {objective_power}")

    return objective_power


class WSI:
    """
    High level class to extract patches from whole slide images; using openslide
    """

    def __init__(self,
                 svs_path,
                 svs_root,
                 save_cache=True,
                 load_cache=False,
                 cache_dir='patches/',
                 mag_ori=None):
        '''
        Args:
            svs_path (str): path to WSI
            save_cache (bool): save cropped patches
            load_cache (bool): load saved caches (patches)
            cache_dir (str): directory to saved patches
        '''

        # path and root directory of WSI
        self.svs_path = svs_path
        self.svs_root = svs_root

        if load_cache:
            pass
        elif mag_ori:
            self.slide = openslide.OpenSlide(os.path.join(svs_root, svs_path))
            self.mag_ori = mag_ori
        else:
            # self.slide is an OpenSlide object representing the slide belonging to the svs_path
            self.slide = openslide.OpenSlide(os.path.join(svs_root, svs_path))
            self.mag_ori = get_original_magnification(self.slide)
            if (self.mag_ori is None):
                raise Exception("[WARNING] Can't find original magnification info from slide, set value in config")

        self.cache_dir = cache_dir
        self.save_cache = save_cache
        self.load_cache = load_cache

    def _extract(self, loc):
        """
        Documentation: https://openslide.org/api/python/
        Args:
            loc(tuple): (x, y) representing top left corner of region to be extracted
        Returns:
            img(PIL.Image): image of patch
        """
        return self.slide.read_region(loc, self.level, self.size)

    def get_multiples(self, xs, ys, size, mag, mag_mask):
        """
        Args:
            xs (list): list of x coordinates of patches to be extracted
            ys (list): list of y coordinates of patches to be extracted
            size (int): size of patch
            mag (int): magnification of patch to be extracted
            mag_mask (int): magnification of mask
        Returns:
            imgs (list): list of PIL.Image objects
        """
        dsf = self.mag_ori / mag
        self.level = self.get_best_level_for_downsample(dsf)
        mag_new = self.mag_ori / (
            [int(x) for x in self.slide.level_downsamples][self.level])
        dsf = mag_new / mag
        dsf_mask = self.mag_ori / mag_mask
        self.size = (int(size * dsf), int(size * dsf))
        xs = tuple(int(x * dsf_mask) for x in xs)
        ys = tuple(int(y * dsf_mask) for y in ys)
        imgs = map(self._extract, list(zip(xs, ys)))
        # for img in imgs:
        #     img.convert('RGB').save(f"tmp2/{uuid.uuid4()}.jpeg")
        return [np.array(img.convert('RGB')) for img in imgs]

    def get_region(self, x, y, size, mag, mag_mask):
        """
        Given the (x, y) location of the region/patch in the thumbnail, this function extracts the patch from the WSI.
        This function proceeds to save the patch in a directory based on the magnification and size of the patch.
        The .jpeg file is stored in nested directories based on the (x, y) location of the region/patch w.r.t. the thumbnail.
        Args:
            x(int): x coordinate of location of patch w.r.t. thumbnail
            y(int): y coordinate of location of patch w.r.t. thumbnail
            size(int): size of patch to be extracted
            mag(int): magnification of patch to be extracted
            mag_mask(int): magnification of mask
        Returns:
            img(np.array): image of patch
            save_dir(str): directory where the patch is saved
        """
        
        svs_id = Path(self.svs_path).stem

        # constructing the save directory where the patches will be saved
        # the jpeg files are saved with names based on their x and y coordinates in the thumbnail
        save_dir = os.path.join(self.cache_dir,
                                f"mag_{str(mag)}-size_{str(size)}", svs_id,
                                f"{x:05d}", f"{y:05d}.jpeg")
        
        # print(f"save_dir: {save_dir}")

        if os.path.isfile(save_dir):
            return None, save_dir

        if self.load_cache:
            img = Image.open(save_dir)
            img = np.array(img)
        else:
            # Calculating downsampling factor for original image coordinates
            dsf = self.mag_ori / mag
            
            # Calculating downsampling factor for original image patch size
            # Value is different because mag_mask may have been resized to lower resolution if requested
            # mag_mask was lower than the available svs downsampling level. ie extracted at 10x but resized to 5x
            level = self.get_best_level_for_downsample(dsf)
            mag_new = self.mag_ori / (
                [int(x) for x in self.slide.level_downsamples][level])
            dsf_new = mag_new / mag
        
            # Passing in the coordinates of the actual WSI from the coords of the thumbnail, the level, and the size of the patch to read_region
            # Documentation of read_region: https://openslide.org/api/python/
            img = self.slide.read_region(
                (int(x * size * dsf), int(y * size * dsf)), level,
                (int(size * dsf_new), int(size * dsf_new)))
            # Converting the image to RGB and resizing it to the desired size
            img = img.convert('RGB').resize((size, size))
            if self.save_cache:
                os.makedirs(os.path.dirname(save_dir), exist_ok=True)
                # saving the image at the save_dir location
                img.save(save_dir)
    
        return np.array(img), save_dir

    def downsample(self, mag):
        """
        This function downsamples the image to the desired magnification
        Args:
            mag (float): magnification to downsample to
        Returns:
            img (np array): downsampled image
        """
        # calculate downsample factor based on original magnification and desired magnification of the patches
        # to reduce the size of the image
        dsf = self.mag_ori / mag

        # gets best level based on the downsample factor calculated above
        level = self.get_best_level_for_downsample(dsf)

        # based on the downsample factor and the best level, calculate the new magnification
        mag_new = self.mag_ori / (
            [int(x) for x in self.slide.level_downsamples][level])

        # calculate the new downsample factor based on the new magnification
        dsf_new = self.mag_ori / mag_new

        # Reads a region of the slide at the selected level and based on the adjusted dimensions. 
        # Uses the read_region function from the self.slide object to retrieve the image region.
        img = self.slide.read_region(
            (0, 0), level,
            tuple(int(x / dsf_new) for x in self.slide.dimensions))

        # sizes is a tuple of the new dimensions of the downsampled image
        sizes = tuple(int(x // dsf) for x in self.slide.dimensions)
        
        # returns the downsampled image as a numpy array in RGB format
        return np.array(img.convert('RGB').resize(sizes))

    def get_best_level_for_downsample(self, factor):
        """
        Gets the best level for a given downsample factor based on properties of the WSI.
        If the factor is greater than the value at the highest level, the highest level is returned
        where the levels are the indices. If the factor is equal to the value at a level, that level is returned.
        If the factor is less than the value at a level, the previous level is returned.

        Args:
            factor: downsample factor
        Returns:
            level: most appropriate level of the slide
        """
        levels = [int(x) for x in self.slide.level_downsamples]

        for i, level in enumerate(levels):
            if factor == level:
                return i
            elif factor > level:
                continue
            elif factor < level:
                return max(i - 1, 0)
        
        return len(levels) - 1