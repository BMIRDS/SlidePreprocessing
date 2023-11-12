from pathlib import Path

import numpy as np
import PIL
from PIL import Image
import openslide

from .wsi import WSI
from .wsi_mask import WsiMask

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
                          cache_dir= str(Path(cache_dir) / study),
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
                                                            threshold=0.2)
            self.positions = pos_left.tolist()
        
        # pos contains up to n coordinates w.r.t. WSI thumbnail. These coordinates
        # represent the location of the patches that have tissue present
        pos = self.positions[(idx * n):(idx * n + n)]
        
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