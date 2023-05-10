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
    fig = plt.figure()
    ax1 = fig.add_axes((0, 0, 1, 1), label='thumbnail')
    ax2 = fig.add_axes((0, 0, 1, 1), label='mask')
    ax1.imshow(thumbnail)
    ax1.axis('off')
    ax2.imshow(mask, alpha=0.5)
    ax2.axis('off')
    plt.savefig(fname, bbox_inches='tight', pad_inches=0)
    plt.close()

    plt.imshow(thumbnail)
    plt.axis('off')
    plt.savefig(fname.replace('-overlay', '-thumbnail'),
                bbox_inches='tight',
                pad_inches=0)
    plt.close()

    plt.imshow(mask)
    plt.axis('off')
    plt.savefig(fname.replace('-overlay', '-mask'),
                bbox_inches='tight',
                pad_inches=0)
    plt.close()


def fix_thumbnail(img):
    size_y, size_x, _ = img.shape
    down_scale = 10 / 0.3125 / 224
    max_x, max_y = int(size_x * down_scale), int(size_y * down_scale)
    new_y, new_x = int(max_x / down_scale), int(max_y / down_scale)
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
    using openslide
    """

    def __init__(self,
                 svs_path,
                 svs_root,
                 save_cache=True,
                 load_cache=False,
                 cache_dir='patches/',
                 mag_ori=None):
        '''
        High level class to extract patches from whole slide images
        svs_path
        save_cache: save cropped patches
        load_cache: load saved caches (patches)
        cache_dir: directory to saved patches
        '''
        self.svs_path = svs_path
        self.svs_root = svs_root

        if load_cache:
            pass
        elif mag_ori:
            self.slide = openslide.OpenSlide(os.path.join(svs_root, svs_path))
            self.mag_ori = mag_ori
        else:
            self.slide = openslide.OpenSlide(os.path.join(svs_root, svs_path))
            self.mag_ori = get_original_magnification(self.slide)
            if (self.mag_ori is None):
                raise Exception("WARNING: Can't find original magnification info from slide, set value in config")
            
        self.cache_dir = cache_dir
        self.save_cache = save_cache
        self.load_cache = load_cache

    def _extract(self, loc):
        return self.slide.read_region(loc, self.level, self.size)

    def get_multiples(self, xs, ys, size, mag, mag_mask):
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
        # TODO: replace with image_extension from config 
        svs_id = self.svs_path.replace('.svs', '')
        svs_id = self.svs_path.replace('.tif', '')
        save_dir = os.path.join(self.cache_dir,
                                f"mag_{str(mag)}-size_{str(size)}", svs_id,
                                f"{x:05d}", f"{y:05d}.jpeg")
        if os.path.isfile(save_dir):
            return None, save_dir

        if self.load_cache:
            img = Image.open(save_dir)
            img = np.array(img)
        else:
            dsf = self.mag_ori / mag
            level = self.get_best_level_for_downsample(dsf)
            mag_new = self.mag_ori / (
                [int(x) for x in self.slide.level_downsamples][level])
            dsf = mag_new / mag
            dsf_mask = self.mag_ori / mag_mask
            img = self.slide.read_region(
                (int(x * dsf_mask), int(y * dsf_mask)), level,
                (int(size * dsf), int(size * dsf)))
            img = img.convert('RGB').resize((size, size))
            if self.save_cache:
                os.makedirs(os.path.dirname(save_dir), exist_ok=True)
                img.save(save_dir)
        return np.array(img), save_dir

    def downsample(self, mag):
        dsf = self.mag_ori / mag
        level = self.get_best_level_for_downsample(dsf)
        mag_new = self.mag_ori / (
            [int(x) for x in self.slide.level_downsamples][level])
        dsf_new = self.mag_ori / mag_new
        img = self.slide.read_region(
            (0, 0), level,
            tuple(int(x / dsf_new) for x in self.slide.dimensions))
        sizes = tuple(int(x // dsf) for x in self.slide.dimensions)
        return np.array(img.convert('RGB').resize(sizes))

    def get_best_level_for_downsample(self, factor):
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
                 mag_mask=0.3125,
                 cache_dir='caches',
                 load_cache=False,
                 saturation_enhance=1,
                 mag_ori=None):
        self.svs_path = svs_path
        self.mag_mask = mag_mask
        self.cache_dir = cache_dir
        self.svs_root = svs_root
        self.study = study
        self.saturation_enhance = saturation_enhance
        self.wsi = WSI(self.svs_path,
                       self.svs_root,
                       load_cache=False,
                       save_cache=False,
                       cache_dir=None,
                       mag_ori=mag_ori)
        self.im_low_res = self.wsi.downsample(self.mag_mask)
        self.load_cache = load_cache
        self.get_mask()

    def sample(self, n, patch_size, mag, threshold, tile_size=None):

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
        patch_cnt = tile_stacked_mask.sum(axis=(2, 3))

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
        patch_locs = patch_locs[:, sel]

        return tile_loc.swapaxes(1, 0).reshape(-1), patch_locs.swapaxes(
            1, 0), (tile_loc + patch_locs).swapaxes(1, 0)

    def get_tissue_map(self, patch_size, mag):
        scale_factor = mag / self.mag_mask
        block_size = int(patch_size / scale_factor)
        h, w = self.mask.shape
        new_h = int(h / (patch_size / scale_factor) * block_size)
        new_w = int(w / (patch_size / scale_factor) * block_size)
        mask = resize(self.mask, (new_h, new_w))
        patch_stacked_mask = view_as_windows(mask, (block_size, block_size),
                                             step=block_size)
        patch_tissue_pct = patch_stacked_mask.mean(axis=(2, 3))
        return patch_tissue_pct

    def sample_all(self, patch_size, mag, threshold):
        patch_tissue_pct = self.get_tissue_map(patch_size, mag)
        patch_mask = patch_tissue_pct > threshold
        patch_mask = remove_small_holes(patch_mask, 10)
        patch_mask = remove_small_objects(patch_mask, 10)
        thumbnail = fix_thumbnail(self.im_low_res.copy())
        combined_view(
            thumbnail, patch_mask,
            self._mask_path(self.svs_path).replace(
                'mask.npy', 'visualization-patch-overlay.jpeg'))
        pos = np.stack(np.where(patch_mask)).swapaxes(1, 0)
        return np.zeros(2).astype(int), pos, pos

    def get_mask(self):
        # get the mask
        mask_file = self._mask_path(self.svs_path)
        if os.path.isfile(mask_file) and self.load_cache:
            self.mask = np.load(mask_file)
        else:
            # print("calculating mask", os.path.basename(self.svs_path))
            self._calculate_mask()
            np.save(mask_file, self.mask)

    def _calculate_mask(self):
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
        mask = filter_composite(thumbnails)
        combined_view(
            thumbnails[0], mask,
            self._mask_path(self.svs_path).replace(
                'mask.npy', 'visualization-tissue-overlay.jpeg'))
        self.mask = mask

    def _mask_path(self, svs_path):
        # TODO: replace with image_extension from config 
        svs_path = svs_path.replace('.svs', '-mask.npy')
        svs_path = svs_path.replace('.tif', '-mask.npy')
        save_path = os.path.join(self.cache_dir, 'masks', self.study,
                                 svs_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        return save_path


class WsiSampler:

    def __init__(self,
                 svs_path,
                 load_cache=False,
                 save_cache=True,
                 cache_dir='patches',
                 svs_root='/pool2/data/WSI_NHCR',
                 study='TCGA',
                 mag_mask=0.3125,
                 saturation_enhance=1,
                 mag_ori=None):
        self.ms = WsiMask(svs_path=svs_path,
                          svs_root=svs_root,
                          study=study,
                          saturation_enhance=saturation_enhance,
                          mag_ori=mag_ori)
        self.mag_mask = mag_mask
        self.svs_path = svs_path
        self.study = study
        self.wsi = WSI(svs_path,
                       svs_root,
                       load_cache=load_cache,
                       save_cache=save_cache,
                       cache_dir=os.path.join(cache_dir, study),
                       mag_ori=mag_ori)
        self.positions = None

    def sample(self, size, n=1, mag=10, tile_size=None):
        pos_tile, pos_l, pos_g = self.ms.sample(n,
                                                size,
                                                mag,
                                                threshold=0.05,
                                                tile_size=tile_size)
        imgs = []
        save_dirs = []
        for pos in pos_g:
            img, save_dir = self.wsi.get_region(pos[1], pos[0], size, mag,
                                                mag / size)
            imgs.append(img)
            save_dirs.append(save_dir)
        return imgs, save_dirs, pos_tile, pos_l, pos_g

    def sample_sequential(self, idx, n, size, mag):
        if self.positions is None:
            self.pos_tile, pos_left, _ = self.ms.sample_all(size,
                                                            mag,
                                                            threshold=0.05)
            self.positions = pos_left.tolist()
        pos = self.positions[(idx * n):(idx * n + n)]

        imgs = []
        save_dirs = []
        for pos_i in pos:
            img, save_dir = self.wsi.get_region(pos_i[1], pos_i[0], size, mag,
                                                mag / size)
            imgs.append(img)
            save_dirs.append(save_dir)
        return imgs, save_dirs, self.pos_tile, pos, pos