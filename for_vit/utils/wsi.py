from typing import Tuple, List, Optional
from pathlib import Path

import numpy as np
import PIL
import PIL.ImageEnhance
from PIL import Image
from skimage.transform import resize
import openslide

def get_original_magnification(slide: openslide.OpenSlide):
    """
    Calculates the original magnification of the given slide, with error handling and calibration.

    Args:
        slide (openslide.OpenSlide): The slide object from which to determine magnification.

    Returns:
        Optional[float]: The original magnification of the slide, calibrated to standard values, or None if not available.
    """
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
    High-level class to extract patches from whole slide images using openslide.

    Attributes:
        svs_path (str): Path to the whole slide image (WSI).
        id_svs (str): Unique identifier for the WSI.
        cache_dir (str): Directory to store cached patches.
        save_cache (bool): Flag to indicate whether to save cropped patches.
        load_cache (bool): Flag to indicate whether to load saved caches (patches).
        slide (openslide.OpenSlide): OpenSlide object representing the WSI.
        mag_original (Optional[float]): Original magnification of the WSI.
    """

    def __init__(self, svs_path: str, id_svs: str, svs_root: str, save_cache: bool = True, 
                 load_cache: bool = False, cache_dir: str = 'patches/', 
                 mag_original: Optional[float] = None) -> None:
        """
        Initializes the WSI object with the specified parameters.

        Args:
            svs_path (str): Path to the WSI file.
            id_svs (str): Unique identifier for the WSI.
            svs_root (str): Root directory for WSI files.
            save_cache (bool): If True, saves cropped patches to cache.
            load_cache (bool): If True, loads patches from cache.
            cache_dir (str): Directory for storing cached patches.
            mag_original (Optional[float]): Original magnification of the WSI, if known.
        """
        '''
        Args:
            svs_path (str): path to WSI
            save_cache (bool): save cropped patches
            load_cache (bool): load saved caches (patches)
            cache_dir (str): directory to saved patches
        '''

        # path and root directory of WSI
        self.svs_path = svs_path
        self.id_svs = id_svs
        self.svs_root = svs_root

        try:
            if load_cache:
                pass
            elif mag_original:
                self.slide = openslide.OpenSlide(str(Path(svs_root) / svs_path))
                self.mag_original = mag_original
            else:
                # self.slide is an OpenSlide object representing the slide belonging to the svs_path
                self.slide = openslide.OpenSlide(str(Path(svs_root) / svs_path))
                self.mag_original = get_original_magnification(self.slide)
                if (self.mag_original is None):
                    raise Exception("[WARNING] Can't find original magnification info from slide, set value in config")
        except openslide.lowlevel.OpenSlideUnsupportedFormatError as e:
            print(f"[ERROR] Failed to open a slide: {str(Path(svs_root) / svs_path)}")
            exit()


        self.cache_dir = cache_dir
        self.save_cache = save_cache
        self.load_cache = load_cache

    def _extract(self, loc: Tuple[int, int]) -> PIL.Image.Image:
        """
        Extracts a region from the WSI based on the given location.

        Args:
            loc (Tuple[int, int]): (x, y) coordinates representing the top left corner of the region.

        Returns:
            PIL.Image.Image: Image of the extracted patch.
        """
        return self.slide.read_region(loc, self.level, self.size)

    def get_multiples(self, x_coords: List[int], y_coords: List[int], patch_size: int, mag: int, 
                      mask_magnification: int) -> List[np.ndarray]:
        """
        Extracts multiple patches from the WSI based on given coordinates, size, and magnification.

        Args:
            x_coords (List[int]): List of x-coordinates of patches.
            y_coords (List[int]): List of y-coordinates of patches.
            patch_size (int): Size of each patch.
            mag (int): Magnification of patches to be extracted.
            mag_mask (int): Magnification of the mask used.

        Returns:
            List[np.ndarray]: List of patches as numpy arrays.
        """
        dsf = self.mag_original / mag
        self.level = self.get_best_level_for_downsample(dsf)
        mag_new = self.mag_original / (
            [int(x) for x in self.slide.level_downsamples][self.level])
        dsf = mag_new / mag
        dsf_mask = self.mag_original / mag_mask
        self.size = (int(patch_size * dsf), int(patch_size * dsf))
        xs = tuple(int(x * dsf_mask) for x in x_coords)
        ys = tuple(int(y * dsf_mask) for y in y_coords)
        imgs = map(self._extract, list(zip(xs, ys)))
        # for img in imgs:
        #     img.convert('RGB').save(f"tmp2/{uuid.uuid4()}.jpeg")
        return [np.array(img.convert('RGB')) for img in imgs]

    def get_region(self, x: int, y: int, patch_size: int, mag: int, mag_mask: int) -> Tuple[np.ndarray, str]:
        """
        Extracts a specific region from the WSI based on coordinates, size, and magnification.

        Args:
            x (int): x-coordinate of the patch's location.
            y (int): y-coordinate of the patch's location.
            patch_size (int): Size of the patch to extract.
            mag (int): Magnification of the patch.
            mag_mask (int): Magnification of the mask used.

        Returns:
            Tuple[np.ndarray, str]: Tuple containing the extracted patch as a numpy array and its save directory.
        """
        
        # constructing the save directory where the patches will be saved
        # the jpeg files are saved with names based on their x and y coordinates in the thumbnail
        
        #TOOD: SHOULD UTILIZE IO_UTILS.PY
        save_path = Path(self.cache_dir) / f"mag_{str(mag)}-size_{str(patch_size)}" / self.id_svs / f"{x:05d}" / f"{y:05d}.jpeg"

        if save_path.exists():
            return None, str(save_path)

        if self.load_cache:
            img = Image.open(str(save_path))
            img = np.array(img)
        else:
            # Calculating downsampling factor for original image coordinates
            dsf = self.mag_original / mag
            
            # Calculating downsampling factor for original image patch size
            # Value is different because mag_mask may have been resized to lower resolution if requested
            # mag_mask was lower than the available svs downsampling level. ie extracted at 10x but resized to 5x
            level = self.get_best_level_for_downsample(dsf)
            mag_new = self.mag_original / (
                [int(x) for x in self.slide.level_downsamples][level])
            dsf_new = mag_new / mag
        
            # Passing in the coordinates of the actual WSI from the coords of the thumbnail, the level, and the size of the patch to read_region
            # Documentation of read_region: https://openslide.org/api/python/
            img = self.slide.read_region(
                (int(x * patch_size * dsf), int(y * patch_size * dsf)), level,
                (int(patch_size * dsf_new), int(patch_size * dsf_new)))
            # Converting the image to RGB and resizing it to the desired size
            img = img.convert('RGB').resize((patch_size, patch_size))
            if self.save_cache:
                Path.mkdir(save_path.parent, parents=True, exist_ok=True)
                
                # saving the image at the save_dir location
                img.save(str(save_path))
    
        return np.array(img), str(save_path)

    def downsample(self, mag_target: float) -> np.ndarray:
        """
        Downsamples the image to a desired magnification level.

        Args:
            mag_target (float): Magnification level to downsample to.

        Returns:
            np.ndarray: Downsampled image as a numpy array.
        """
        # calculate downsample factor based on original magnification and desired magnification of the patches
        # to reduce the size of the image
        dsf = self.mag_original / mag_target

        # gets best level based on the downsample factor calculated above
        level = self.get_best_level_for_downsample(dsf)

        # based on the downsample factor and the best level, calculate the new magnification
        mag_new = self.mag_original / (
            [int(x) for x in self.slide.level_downsamples][level])

        # calculate the new downsample factor based on the new magnification
        dsf_new = self.mag_original / mag_new

        # Reads a region of the slide at the selected level and based on the adjusted dimensions. 
        # Uses the read_region function from the self.slide object to retrieve the image region.
        img = self.slide.read_region(
            (0, 0), level,
            tuple(int(x / dsf_new) for x in self.slide.dimensions))

        # sizes is a tuple of the new dimensions of the downsampled image
        sizes = tuple(int(x // dsf) for x in self.slide.dimensions)
        
        # returns the downsampled image as a numpy array in RGB format
        return np.array(img.convert('RGB').resize(sizes))

    def get_best_level_for_downsample(self, downsample_factor: float) -> int:
        """
        Determines the best level for downsampling based on the given factor.

        Args:
            downsample_factor (float): The factor by which to downsample the image.

        Returns:
            int: The most appropriate level for downsampling.
        """
        levels = [int(x) for x in self.slide.level_downsamples]

        for i, level in enumerate(levels):
            if downsample_factor == level:
                return i
            elif downsample_factor > level:
                continue
            elif downsample_factor < level:
                return max(i - 1, 0)
        
        return len(levels) - 1