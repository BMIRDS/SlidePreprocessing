import os
import numpy as np
import PIL
import PIL.ImageEnhance
from PIL import Image
from skimage.util.shape import view_as_windows
from skimage.morphology import \
    (remove_small_objects, remove_small_holes, binary_erosion, binary_dilation)
from skimage.transform import resize
import openslide
import matplotlib.pyplot as plt
from .filter import filter_composite


def combined_view(thumbnail, mask, fname):
    """
    This function combines the thumbnail and mask images and saves the figure.
    The function also saves the thumbnail and mask images separately.
    
    Args:
        thumbnail(np.array): thumbnail image
        mask(np.array): mask image (binary)
        fname(str): file name to save the figure
    Returns:
        None
    """

    # plotting the visualization of mask overlay on thumbnail
    fig = plt.figure()
    ax1 = fig.add_axes((0, 0, 1, 1), label='thumbnail')
    ax2 = fig.add_axes((0, 0, 1, 1), label='mask')
    ax1.imshow(thumbnail)
    ax1.axis('off')
    ax2.imshow(mask, alpha=0.5)
    ax2.axis('off')
    plt.savefig(fname, bbox_inches='tight', pad_inches=0)
    plt.close()

    # plotting the thumnail
    plt.imshow(thumbnail)
    plt.axis('off')
    plt.savefig(fname.replace('-overlay', '-thumbnail'),
                bbox_inches='tight',
                pad_inches=0)
    plt.close()

    # plotting the mask
    plt.imshow(mask)
    plt.axis('off')
    plt.savefig(fname.replace('-overlay', '-mask'),
                bbox_inches='tight',
                pad_inches=0)
    plt.close()


def fix_thumbnail(img, mag, mag_mask, patch_size):
    """
    This function cropts the img and returns it
    Args:
        img(np.array): input image to be resized and cropped
    Returns:
        img(np.array): adjusted cropped image
    """

    # getting width, height dimensions of image
    size_y, size_x, _ = img.shape

    # calculating the downscale factor based on magnification, mag_mask, and patch_size
    down_scale = mag / mag_mask / patch_size
    max_x, max_y = int(size_x * down_scale), int(size_y * down_scale)
    new_y, new_x = int(max_x / down_scale), int(max_y / down_scale)

    # cropts the image
    img = img[:new_x, :new_y, :]

    return img

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
                raise Exception("WARNING: Can't find original magnification info from slide, set value in config")

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
        # TODO: replace with image_extension from config 
        svs_id = self.svs_path.replace('.svs', '')
        svs_id = self.svs_path.replace('.tif', '')

        # print(f"svs_id: {svs_id}, x: {x}, y: {y}, size: {size}, mag: {mag}, mag_mask: {mag_mask}")
        
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
            # Calculating downsampling factors, level, etc.
            dsf = self.mag_ori / mag
            # print(f"dsf with mag_ori: {dsf}")
            level = self.get_best_level_for_downsample(dsf)
            # print(f"level: {level}")
            mag_new = self.mag_ori / (
                [int(x) for x in self.slide.level_downsamples][level])
            dsf = mag_new / mag
            dsf_mask = self.mag_ori / mag_mask
            
            #block size of patches in mask
            block_size = int(size / dsf_mask)

            # print(f"Parameters being passed into slide_region: {(int(x * dsf_mask), int(y * dsf_mask)), level, (int(size * dsf), int(size * dsf))}")
            # Passing in the coordinates of the actual WSI from the coords of the thumbnail, the level, and the size of the patch to read_region
            # Documentation of read_region: https://openslide.org/api/python/
            img = self.slide.read_region(
                (int(x * dsf_mask * block_size), int(y * dsf_mask * block_size)), level,
                (int(size * dsf), int(size * dsf)))
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


class WsiMask:
    '''
    get the mask map of a given WSI
    '''

    def __init__(self,
                 svs_path='',
                 svs_root='',
                 study='',
                 mag_mask=None,
                 cache_dir='caches',
                 load_cache=False,
                 saturation_enhance=1,
                 mag_ori=None,
                 filtering_style=''):
        self.svs_path = svs_path
        self.cache_dir = cache_dir
        self.svs_root = svs_root
        self.study = study
        self.saturation_enhance = saturation_enhance
        self.filtering_style = filtering_style
        self.wsi = WSI(self.svs_path,
                       self.svs_root,
                       load_cache=False,
                       save_cache=False,
                       cache_dir=None,
                       mag_ori=mag_ori)
        self.mag_mask = self.fix_mag_mask(mag_mask)
        self.im_low_res = self.wsi.downsample(self.mag_mask)
        self.load_cache = load_cache
        self.get_mask()
        
    def fix_mag_mask(self, mag_mask):
        """
        This function makes mask magnification compatible with available downsample levels
        Args:
            mag_mask (float): ideal mask magnification
        Returns:
            mag_new (float): closest possible mask magnification to input
        """
        dsf = self.wsi.mag_ori / mag_mask
        
        level = self.wsi.get_best_level_for_downsample(dsf)
        
        mag_new = self.wsi.mag_ori / (
            [int(x) for x in self.wsi.slide.level_downsamples][level])
        
        return mag_new
        
    def sample(self, n, patch_size, mag, threshold, tile_size=None):
        """
        TODO: Implementation of get_topk_threshold function
        Args:
            n (int): number of patches to sample
            patch_size (int): size of the patch
            mag (float): magnification to sample patches at
            threshold (float): threshold for tissue in a patch
            tile_size (int): size of the tile
        Returns:
            tile_loc (np array): TBD
            patch_loc (np array): TBD
            (np array): TBD
        """

        if tile_size is None:
            pos_tile, pos_l, _ = self.sample_all(patch_size, mag, threshold)
            pos_l = pos_l[np.random.choice(pos_l.shape[0], n)]
            return pos_tile, pos_l, pos_l

        patch_tissue_pct = self.get_tissue_map(patch_size, mag)
        patch_tissue_mask = (patch_tissue_pct > threshold).astype(int)

        tile_size = int(tile_size / patch_size)
        tile_size_corrected = min(tile_size, *patch_tissue_mask.shape)
        tile_stacked_mask = view_as_windows(
            patch_tissue_mask, (tile_size_corrected, tile_size_corrected),
            step=1)
        # determining the patch count for each tile
        patch_cnt = tile_stacked_mask.sum(axis=(2, 3))

        # selecting a tile with a sufficient patch count
        tile_locs = np.stack(
            np.where(patch_cnt >= min(n, get_topk_threshold(patch_cnt, 20))))
        tile_loc = tile_locs[:,
                             np.random.choice(tile_locs.shape[1], 1)].reshape(
                                 2, 1)
        tile_x, tile_y = tile_loc.reshape(-1)

        tile = patch_tissue_pct[tile_x:(tile_x + tile_size),
                                tile_y:(tile_y + tile_size)]
        patch_locs = np.stack(
            np.where(tile >= min(threshold, get_topk_threshold(tile, n))))

        sel = np.random.choice(patch_locs.shape[1],
                               n,
                               replace=n > patch_locs.shape[1])
        # stores the patch locations
        patch_locs = patch_locs[:, sel]

        return tile_loc.swapaxes(1, 0).reshape(-1), patch_locs.swapaxes(
            1, 0), (tile_loc + patch_locs).swapaxes(1, 0)

    def get_tissue_map(self, patch_size, mag):
        """
        Get a "map" where each entry corresponds to a tissue percentage in a patch of the original WSI.

        Args:
            patch_size (int): the size of the patch
            mag (float): the magnification of the patch
        Returns:
            patch_tissue_pct (2D numpy array): a "map" where each entry corresponds to a tissue percentage in a patch of the original WSI.
        """
        #Sets standard mag_mask value if one has not been passed in init
        if self.mag_mask == None:
            self.mag_mask = self.fix_mag_mask(mag / patch_size)
            print("[WARNING] No mag_mask value passed to WsiMask initializer, setting to ", self.mag_mask, " based off mag/patch_size")
            
        # ratio of reqested magnification level of patch extraction to the magnification level of the mask
        scale_factor = mag / self.mag_mask

        # block_size represents the size of the patches in the binary tissue mask
        # block_size * scale_factor = size of patch in the requested magnification level
        block_size = int(patch_size / scale_factor)
        h, w = self.mask.shape
        # new_h and new_w represent the dimensions of the resized mask to the scale_factor
        new_h = int(h / (patch_size / scale_factor) * block_size)
        new_w = int(w / (patch_size / scale_factor) * block_size)
        # resize the mask to the new dimensions
        mask = resize(self.mask, (new_h, new_w))
        patch_stacked_mask = view_as_windows(mask, (block_size, block_size),
                                             step=block_size)
        # patch_tissue_pct is the percentage of tissue in each patch
        patch_tissue_pct = patch_stacked_mask.mean(axis=(2, 3))

        return patch_tissue_pct

    def sample_all(self, patch_size, mag, threshold):
        """
        Args:
            patch_size (int): the size of the patch
            mag (float): the magnification of the patch
            threshold (float): the threshold of the tissue
        
        Returns:
            numpy array: stores zeroes (TODO: possible placeholder)
            pos (numpy array): stores the coordinates of patches where tissue is present in the WSI thumnail
        """

        # path_tissue_pct represents a 'map' where each entry corresponds to a tissue percentage in a patch of the original WSI
        
        patch_tissue_pct = self.get_tissue_map(patch_size, mag)

        # patch_mask represents a binary 'map' based on the threshold
        patch_mask = patch_tissue_pct > threshold

        # smoothing out the masks
        patch_mask = remove_small_holes(patch_mask, 10)
        patch_mask = remove_small_objects(patch_mask, 10)

        # adjsts the thumbnail of the WSI which has been downsampled to mag_mask
        thumbnail = fix_thumbnail(self.im_low_res.copy(), mag, self.mag_mask, patch_size)
        
        # saving images of the overlay the mask on the thumbnail
        combined_view(
            thumbnail, patch_mask,
            self._mask_path(self.svs_path).replace(
                'mask.npy', 'visualization-patch-overlay.jpeg'))

        # list of pixel coordinates w.r.t. thumnail image of the patches where the mask is true (where tissue is present)
        pos = np.stack(np.where(patch_mask)).swapaxes(1, 0)

        return np.zeros(2).astype(int), pos

    def get_mask(self):
        """
        Gets the mask of the WSI specified by svs_path. If the mask exists, load the mask. Otherwise, calculate the mask and save it.
        """
        # get the mask
        mask_file = self._mask_path(self.svs_path)
        # if mask exists, load the mask
        if os.path.isfile(mask_file) and self.load_cache:
            self.mask = np.load(mask_file)
        else:
            # print("calculating mask", os.path.basename(self.svs_path))
            self._calculate_mask()
            np.save(mask_file, self.mask)

    def _calculate_mask(self):
        """
        Calculates the mask of the WSI and saves it in an instance variable.
        """
        thumbnails = [self.im_low_res.copy()]
        if self.saturation_enhance == 1:
            pass
        else:
            for saturation_enhance in [0.5, 2, 4]:
                thumbnail = self.im_low_res.copy()
                converter = PIL.ImageEnhance.Color(
                    PIL.Image.fromarray(thumbnail))
                thumbnail = converter.enhance(saturation_enhance)
                thumbnail = np.array(thumbnail)
                thumbnails.append(thumbnail)
        # mask is a binary composite mask that selects regions with colors
        mask = filter_composite(thumbnails, self.filtering_style)
        # saving the images of the overlay of the map
        combined_view(
            thumbnails[0], mask,
            self._mask_path(self.svs_path).replace(
                'mask.npy', 'visualization-tissue-overlay.jpeg'))
        self.mask = mask

    def _mask_path(self, svs_path):
        """
        Args:
            svs_path (str): the path to the WSI
        Returns:
            str: the path to where the mask file of the WSI is stored
        """
        # TODO: replace with image_extension from config 
        svs_path = svs_path.replace('.svs', '-mask.npy')
        svs_path = svs_path.replace('.tif', '-mask.npy')
        save_path = os.path.join(self.cache_dir, 'masks', self.study,
                                 svs_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        return save_path


class WsiSampler:

    def __init__(self,
                 svs_path='',
                 load_cache=False,
                 save_cache=True,
                 cache_dir='patches',
                 svs_root='',
                 study='',
                 mag_mask=None,
                 saturation_enhance=1,
                 mag_ori=None,
                 filtering_style=''):
        self.wsi = WSI(svs_path,
                          svs_root,
                          load_cache=load_cache,
                          save_cache=save_cache,
                          cache_dir=os.path.join(cache_dir, study),
                          mag_ori=mag_ori)
        self.ms = WsiMask(svs_path=svs_path,
                          svs_root=svs_root,
                          study=study,
                          mag_mask=mag_mask,
                          saturation_enhance=saturation_enhance,
                          mag_ori=mag_ori,
                          filtering_style=filtering_style)
        self.mag_mask = self.ms.mag_mask
        self.svs_path = svs_path
        self.study = study
        self.positions = None

    def sample(self, size, n=1, mag=10, tile_size=None):
        """
        Args:
            size (int): the size of the patch
            n (int): the number of patches to sample
            mag (float): the magnification of the patch to sample
            tile_size (int): the size of the tile to sample
        Returns:
            imgs (list): a list of images (np array) of the patches
            save_dirs (list): a list of the paths to where the patches are saved
            pos_tile (list): TBD
            pos_l (list): TBD
            pos_g (list): TBD

        """
        pos_tile, pos_l, pos_g = self.ms.sample(n,
                                                size,
                                                mag,
                                                threshold=0.05,
                                                tile_size=tile_size)
        # print(f"pos_tile: {pos_tile} pos_l: {pos_l} pos_g: {pos_g}")
        imgs = []
        save_dirs = []
        for pos in pos_g:
            img, save_dir = self.wsi.get_region(pos[1], pos[0], size, mag,
                                                mag / size)
            imgs.append(img)
            save_dirs.append(save_dir)
        return imgs, save_dirs, pos_tile, pos_l, pos_g

    def sample_sequential(self, idx, n, patch_size, mag):
        """
        This function is the main driver behind storing coordinates of the
        regions/patches that have tissue and calling functions to extract and save
        such patches from the WSI
        
        Args:
            idx (int): index of the batch
            n (int): number of patches per batch
            patch_size (int): size of the patch
            mag (int): magnification of the patch
        Returns:
            imgs (list): list of images (np arrays)
            save_dirs (list): list of save directories
        """
        if self.positions is None:
            self.pos_tile, pos_left = self.ms.sample_all(patch_size,
                                                            mag,
                                                            threshold=0.25)
            self.positions = pos_left.tolist()
        
        # pos contains up to n coordinates w.r.t. WSI thumbnail. These coordinates
        # represent the location of the patches that have tissue present
        pos = self.positions[(idx * n):(idx * n + n)]
        
        #Sets standard mag_mask value if one has not been passed in init
        if self.mag_mask == None:
            self.mag_mask = self.ms.fix_mag_mask(mag / patch_size)
            print("[WARNING] No mag_mask value passed to WsiSampler initializer, setting to ", self.mag_mask, " based off mag/patch_size")

        imgs = []
        save_dirs = []
        for pos_i in pos:
            # start the process of extracting the patch from the WSI
            img, save_dir = self.wsi.get_region(pos_i[1], pos_i[0], patch_size, mag,
                                                self.mag_mask)
            # add the images to imgs list and the save_dir to save_dirs list
            imgs.append(img)
            save_dirs.append(save_dir)
        return imgs, save_dirs, self.pos_tile, pos, pos