from pathlib import Path

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

from .wsi import WSI
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
        self.mag_mask = mag_mask
        self.im_low_res = self.wsi.downsample(self.mag_mask)
        self.load_cache = load_cache
        self.get_mask()
        
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
        # scale factor is the downscaling ratio to the mask
        scale_factor = mag / self.mag_mask

        # block_size represents the size of the patches(factoring in overall downsampling) in the binary tissue mask
        block_size = int(patch_size / scale_factor)
        
        assert (block_size >= 1), "Mask Mag too low to extract a tissue map for requested patch size. Increase mag_mask or patch_size in config"
        
        h, w = self.mask.shape
        # new_h and new_w represent the dimensions of the resized mask to the scale_factor
        new_h = int(h / (patch_size / scale_factor) * block_size)
        new_w = int(w / (patch_size / scale_factor) * block_size)
        
        # resize the mask to the new dimensions
        mask = resize(self.mask, (new_h, new_w))
        patch_stacked_mask = view_as_windows(self.mask, (block_size, block_size),
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
        if Path(mask_file).exists() and self.load_cache:
            self.mask = np.load(mask_file)
        else:
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
        mask_path = f"{Path(svs_path).stem}{'-mask.npy'}"
        
        save_path = Path(self.cache_dir) / "masks" / self.study / mask_path
        
        Path.mkdir(save_path.parent, parents=True, exist_ok=True)
        
        return str(save_path)
  
  