from pathlib import Path
from typing import List, Tuple

import numpy as np
import PIL
from PIL import Image
import openslide

from .wsi import WSI
from .wsi_mask import WsiMask

class WsiSampler:
    """
    A sampler class for Whole Slide Images (WSIs) that facilitates the extraction and processing of image patches.

    Attributes:
        wsi (WSI): An instance of the WSI class for handling whole slide image operations.
        ms (WsiMask): An instance of the WsiMask class for handling mask-related operations in WSIs.
        mag_mask (float): Magnification level of the mask.
        svs_path (str): Path to the .svs file.
        study_identifier (str): Identifier for the study.
        id_svs (str): Identifier for the WSI.
        positions (list): List of positions for sequential sampling.

    Methods:
        sample: Samples a specified number of patches from the WSI at a given magnification.
        sample_sequential: Sequentially samples patches from pre-determined positions in the WSI.
    """

    def __init__(self, svs_path: str, id_svs: str, study_identifier: str, 
                 load_cache: bool = False, save_cache: bool = True, 
                 cache_dir: str = 'patches', svs_root: str = '', 
                 mag_mask: float = None, saturation_enhance: float = 1, 
                 mag_original: float = None, filter_style: str = '') -> None:
        """
        Initializes the WsiSampler with the specified parameters and creates instances of WSI and WsiMask.

        Args:
            svs_path (str): Required. Path to the .svs file for the WSI.
            id_svs (str): Required. Unique identifier for the WSI.
            study_identifier (str): Required. Identifier for the study.
            load_cache (bool): Flag to load cached data if available.
            save_cache (bool): Flag to save processed data to cache.
            cache_dir (str): Directory for storing cached data.
            svs_root (str): Root directory for .svs files.
            mag_mask (float): Magnification level for the mask.
            saturation_enhance (float): Factor for saturation enhancement.
            mag_original (float): Original magnification of the WSI.
            filter_style (str): Style of filtering applied to the WSI.
        """

        self.wsi = WSI(svs_path,
                        id_svs,
                        svs_root,
                        load_cache=load_cache,
                        save_cache=save_cache,
                        cache_dir= str(Path(cache_dir) / study_identifier),
                        mag_original=mag_original)
        self.ms = WsiMask(svs_path=svs_path,
                          svs_root=svs_root,
                          id_svs=id_svs,
                          study_identifier=study_identifier,
                          mag_mask=mag_mask,
                          saturation_enhance=saturation_enhance,
                          mag_original=mag_original,
                          filter_style=filter_style)
        self.mag_mask = self.ms.mag_mask
        self.svs_path = svs_path
        # self.study_identifier = study_identifier
        self.positions = None

    def sample(self, patch_size: int, num_patches: int = 1, mag: float = 10, 
               tile_size: int = None) -> Tuple[List[np.ndarray], List[str], List[Tuple[int, int]], 
                                               List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Samples a specified number of patches from the WSI at a given magnification and size.

        Args:
            size (int): Size of the patch to sample.
            num_patches (int): Number of patches to sample.
            mag (float): Magnification level for sampling.
            tile_size (int, optional): Size of the tile for sampling.

        Returns:
            list: Images of sampled patches.
            list: Save directories for each sampled patch.
            list: Positions of sampled patches in tile coordinates.
            list: Positions of sampled patches in local coordinates.
            list: Positions of sampled patches in global coordinates.
        """
        pos_tile, pos_l, pos_g = self.ms.sample(num_patches,
                                                patch_size,
                                                mag,
                                                threshold=0.05,
                                                tile_size=tile_size)
        # print(f"pos_tile: {pos_tile} pos_l: {pos_l} pos_g: {pos_g}")
        imgs = []
        save_dirs = []
        for pos in pos_g:
            img, save_dir = self.wsi.get_region(pos[1], pos[0], patch_size, mag,
                                                mag / patch_size)
            imgs.append(img)
            save_dirs.append(save_dir)
        return imgs, save_dirs, pos_tile, pos_l, pos_g

    def sample_sequential(self, batch_index: int, num_patches: int, patch_size: int, 
                          mag: int) -> Tuple[List[np.ndarray], List[str], List[Tuple[int, int]], 
                                                       List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Sequentially samples patches from predetermined positions in the WSI.

        Args:
            batch_index (int): Index of the batch to sample.
            num_patches (int): Number of patches to sample per batch.
            patch_size (int): Size of each patch.
            mag (int): Magnification level for sampling.

        Returns:
            Tuple: Containing lists of images of sampled patches, save directories for each patch, and positions.
        """

        if self.positions is None:
            self.pos_tile, pos_left = self.ms.sample_all(patch_size,
                                                            mag,
                                                            threshold=0.2)
            self.positions = pos_left.tolist()
        
        # pos contains up to n coordinates w.r.t. WSI thumbnail. These coordinates
        # represent the location of the patches that have tissue present
        start_index = batch_index * num_patches
        pos = self.positions[(start_index):(start_index + num_patches)]
        
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
