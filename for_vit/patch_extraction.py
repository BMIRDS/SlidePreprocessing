import os
import pandas as pd
import numpy as np
import PIL
import PIL.ImageEnhance
from PIL import Image
from skimage.util.shape import view_as_windows
from skimage.morphology import \
    (remove_small_objects, remove_small_holes, binary_erosion, binary_dilation)
from skimage.transform import resize
import openslide
from tqdm.contrib.concurrent import process_map
import argparse

parser = argparse.ArgumentParser(description='Patch extraction')
parser.add_argument('-c',
                    '--cancer',
                    type=str,
                    default='TCGA_BLCA',
                    help='cancer subset')
parser.add_argument('-j', '--num-workers', type=int, default=10)
parser.add_argument('-m', '--magnification', type=int, default=10)
parser.add_argument('-s', '--patch-size', type=int, default=224)
parser.add_argument('--svs-meta', type=str, default='meta/dhmc_rcc_svs.pickle')
args = parser.parse_args()


class WSI:
    """
    using openslide
    """

    def __init__(self,
                 svs_path,
                 svs_root,
                 save_cache=True,
                 load_cache=False,
                 cache_dir='patches/'):
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
        else:
            self.slide = openslide.OpenSlide(os.path.join(svs_root, svs_path))
            self.mag_ori = int(
                float(self.slide.properties.get('aperio.AppMag', 40)))
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
        svs_id = self.svs_path.replace('.svs', '')
        save_dir = os.path.join(self.cache_dir,
                                f"mag_{str(mag)}-size_{str(size)}", svs_id,
                                f"{x:05d}", f"{y:05d}.jpeg")
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
        return np.array(img)

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


def get_topk_threshold(arr, k):
    arr = arr.reshape(-1)
    if k >= len(arr):
        return arr.min()
    temp = np.argpartition(-arr, k)
    temp = np.partition(-arr, k)
    result = -temp[:k]
    return result.min()


def is_purple_dot(r, g, b):
    rb_avg = (r + b) / 2
    if r > g - 10 and b > g - 10 and rb_avg > g + 20:
        return True
    return False


SIZE = 8


def is_purple(crop):
    crop = crop.reshape(SIZE, SIZE, 3)
    for x in range(crop.shape[0]):
        for y in range(crop.shape[1]):
            r = crop[x, y, 0]
            g = crop[x, y, 1]
            b = crop[x, y, 2]
            if is_purple_dot(r, g, b):
                return True
    return False


def filter_purple(img):
    h, w, d = img.shape
    step = SIZE
    img_padding = np.zeros((h + step - 1, w + step - 1, d))
    img_padding[:h, :w, :d] = img
    img_scaled = view_as_windows(img_padding, (SIZE, SIZE, 3), 1)
    return np.apply_along_axis(is_purple, -1,
                               img_scaled.reshape(h, w, -1)).astype(int)


def filter_grays(rgb, tolerance=15, output_type="bool"):
    """
  Create a mask to filter out pixels where the red, green, and blue channel values are similar.
  Args:
    np_img: RGB image as a NumPy array.
    tolerance: Tolerance value to determine how similar the values must be in order to be filtered out
    output_type: Type of array to return (bool, float, or uint8).
  Returns:
    NumPy array representing a mask where pixels with similar red, green, and blue values have been masked out.
  """
    (h, w, c) = rgb.shape

    rgb = rgb.astype(np.int)
    rg_diff = abs(rgb[:, :, 0] - rgb[:, :, 1]) <= tolerance
    rb_diff = abs(rgb[:, :, 0] - rgb[:, :, 2]) <= tolerance
    gb_diff = abs(rgb[:, :, 1] - rgb[:, :, 2]) <= tolerance
    result = ~(rg_diff & rb_diff & gb_diff)
    return result


def filter_blue(rgb, red_upper_thresh, green_upper_thresh, blue_lower_thresh):
    """
    credit: https://github.com/CODAIT/deep-histopath/blob/c8baf8d47b6c08c0f6c7b1fb6d5dd6b77e711c33/deephistopath/wsi/filter.py#L771
    """
    r = rgb[:, :, 0] < red_upper_thresh
    g = rgb[:, :, 1] < green_upper_thresh
    b = rgb[:, :, 2] > blue_lower_thresh
    result = ~(r & g & b)
    return result


def filter_blue_pen(rgb):
    result = filter_blue(rgb, red_upper_thresh=60, green_upper_thresh=120, blue_lower_thresh=190) & \
           filter_blue(rgb, red_upper_thresh=120, green_upper_thresh=170, blue_lower_thresh=200) & \
           filter_blue(rgb, red_upper_thresh=175, green_upper_thresh=210, blue_lower_thresh=230) & \
           filter_blue(rgb, red_upper_thresh=145, green_upper_thresh=180, blue_lower_thresh=210) & \
           filter_blue(rgb, red_upper_thresh=37, green_upper_thresh=95, blue_lower_thresh=160) & \
           filter_blue(rgb, red_upper_thresh=30, green_upper_thresh=65, blue_lower_thresh=130) & \
           filter_blue(rgb, red_upper_thresh=130, green_upper_thresh=155, blue_lower_thresh=180) & \
           filter_blue(rgb, red_upper_thresh=40, green_upper_thresh=35, blue_lower_thresh=85) & \
           filter_blue(rgb, red_upper_thresh=30, green_upper_thresh=20, blue_lower_thresh=65) & \
           filter_blue(rgb, red_upper_thresh=90, green_upper_thresh=90, blue_lower_thresh=140) & \
           filter_blue(rgb, red_upper_thresh=60, green_upper_thresh=60, blue_lower_thresh=120) & \
           filter_blue(rgb, red_upper_thresh=110, green_upper_thresh=110, blue_lower_thresh=175)
    return result


def filter_green(rgb, red_upper_thresh, green_lower_thresh, blue_lower_thresh):
    r = rgb[:, :, 0] < red_upper_thresh
    g = rgb[:, :, 1] > green_lower_thresh
    b = rgb[:, :, 2] > blue_lower_thresh
    result = ~(r & g & b)
    return result


def filter_green_pen(rgb):
    result = filter_green(rgb, red_upper_thresh=150, green_lower_thresh=160, blue_lower_thresh=140) & \
           filter_green(rgb, red_upper_thresh=70, green_lower_thresh=110, blue_lower_thresh=110) & \
           filter_green(rgb, red_upper_thresh=45, green_lower_thresh=115, blue_lower_thresh=100) & \
           filter_green(rgb, red_upper_thresh=30, green_lower_thresh=75, blue_lower_thresh=60) & \
           filter_green(rgb, red_upper_thresh=195, green_lower_thresh=220, blue_lower_thresh=210) & \
           filter_green(rgb, red_upper_thresh=225, green_lower_thresh=230, blue_lower_thresh=225) & \
           filter_green(rgb, red_upper_thresh=170, green_lower_thresh=210, blue_lower_thresh=200) & \
           filter_green(rgb, red_upper_thresh=20, green_lower_thresh=30, blue_lower_thresh=20) & \
           filter_green(rgb, red_upper_thresh=50, green_lower_thresh=60, blue_lower_thresh=40) & \
           filter_green(rgb, red_upper_thresh=30, green_lower_thresh=50, blue_lower_thresh=35) & \
           filter_green(rgb, red_upper_thresh=65, green_lower_thresh=70, blue_lower_thresh=60) & \
           filter_green(rgb, red_upper_thresh=100, green_lower_thresh=110, blue_lower_thresh=105) & \
           filter_green(rgb, red_upper_thresh=165, green_lower_thresh=180, blue_lower_thresh=180) & \
           filter_green(rgb, red_upper_thresh=140, green_lower_thresh=140, blue_lower_thresh=150) & \
           filter_green(rgb, red_upper_thresh=185, green_lower_thresh=195, blue_lower_thresh=195)
    return result


def filter_composite(img):
    # select the region with colors
    # mask_s = matplotlib.colors.rgb_to_hsv(img)[:,:,1] > 0.05
    mask_s = filter_purple(img)
    # mask_s = binary_erosion(binary_dilation(mask_s, np.ones((10,10))), np.ones((10,10)))
    # filter pen marks
    mask_no_pen = filter_blue_pen(img) & filter_green_pen(img)
    mask = mask_s & mask_no_pen
    # mask = remove_small_holes(mask, 1000)
    mask = remove_small_objects(mask > 0, 100)
    return mask


def combined_view(thumbnail, mask, fname):
    _mask = Image.fromarray(mask.astype(np.uint8) * 255)
    _thumbnail = Image.fromarray(thumbnail.astype(np.uint8))
    new_im = Image.new('RGB', (_mask.size[0] + _thumbnail.size[0],
                               max(_mask.size[1], _thumbnail.size[1])))

    x_offset = 0
    for im in [_mask, _thumbnail]:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]
    new_im.save(fname)


class WsiMask:
    '''
    get the mask map of a given WSI
    '''

    def __init__(self,
                 svs_path='',
                 svs_root='',
                 study='',
                 mag_mask=0.25,
                 cache_dir='caches',
                 saturation_enhance=1):
        self.svs_path = svs_path
        self.mag_mask = mag_mask
        self.cache_dir = cache_dir
        self.svs_root = svs_root
        self.study = study
        self.saturation_enhance = saturation_enhance
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
        pos = np.stack(np.where(patch_tissue_pct > threshold)).swapaxes(1, 0)
        return np.zeros(2).astype(int), pos, pos

    def get_mask(self):
        # get the mask
        mask_file = self._mask_path(self.svs_path)
        if os.path.isfile(mask_file):
            self.mask = np.load(mask_file)
        else:
            print("calculating mask", os.path.basename(self.svs_path))
            self._calculate_mask()
            np.save(mask_file, self.mask)

    def _calculate_mask(self):
        wsi = WSI(self.svs_path,
                  self.svs_root,
                  load_cache=False,
                  save_cache=False,
                  cache_dir=None)
        im_low_res = wsi.downsample(self.mag_mask)
        if self.saturation_enhance == 1:
            pass
        else:
            converter = PIL.ImageEnhance.Color(PIL.Image.fromarray(im_low_res))
            im_low_res = converter.enhance(self.saturation_enhance)
            im_low_res = np.array(im_low_res)
        mask = filter_composite(im_low_res)
        combined_view(
            im_low_res, mask,
            self._mask_path(self.svs_path).replace('mask.npy',
                                                   'visualization.jpeg'))
        self.mask = mask

    def _mask_path(self, svs_path):
        save_path = os.path.join(self.cache_dir, 'masks', self.study,
                                 svs_path.replace('.svs', '-mask.npy'))
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
                 mag_mask=0.25,
                 saturation_enhance=1):
        self.ms = WsiMask(svs_path=svs_path,
                          svs_root=svs_root,
                          study=study,
                          saturation_enhance=saturation_enhance)
        self.mag_mask = mag_mask
        self.svs_path = svs_path
        self.study = study
        self.wsi = WSI(svs_path,
                       svs_root,
                       load_cache=load_cache,
                       save_cache=save_cache,
                       cache_dir=os.path.join(cache_dir, study))
        self.positions = None

    def sample(self, size, n=1, mag=10, tile_size=None):
        pos_tile, pos_l, pos_g = self.ms.sample(n,
                                                size,
                                                mag,
                                                threshold=0.5,
                                                tile_size=tile_size)
        imgs = []
        for pos in pos_g:
            imgs.append(
                self.wsi.get_region(pos[1], pos[0], size, mag, mag / size))
        return imgs, pos_tile, pos_l, pos_g

    def sample_sequential(self, idx, n, size, mag):
        if self.positions is None:
            self.pos_tile, pos_left, _ = self.ms.sample_all(size,
                                                            mag,
                                                            threshold=0.5)
            self.positions = pos_left.tolist()
        pos = self.positions[(idx * n):(idx * n + n)]

        imgs = []
        for pos_i in pos:
            imgs.append(
                self.wsi.get_region(pos_i[1], pos_i[0], size, mag, mag / size))
        return imgs, self.pos_tile, pos, pos


def get_masks(inputs):
    svs_path, svs_root, study = inputs
    try:
        WsiSampler(svs_path=svs_path,
                   svs_root=svs_root,
                   study=study,
                   saturation_enhance=2)
    except Exception as e:
        print(svs_path, e)


def get_patches(inputs):
    svs_path, svs_root, study = inputs
    try:
        wsi = WsiSampler(svs_path=svs_path, svs_root=svs_root, study=study)
        _ = wsi.sample_sequential(0, 100000, args.patch_size,
                                  args.magnification)
    except Exception as e:
        print(svs_path, e)


if __name__ == '__main__':

    df_sub = pd.read_pickle(args.svs_meta)
    paired_inputs = []
    for i, row in df_sub.iterrows():
        if row['id_svs'] in [
                'TCGA-UZ-A9PQ-01Z-00-DX1.C2CB0E94-2548-4399-BCAB-E4D556D533EF',
                'TCGA-5P-A9KA-01Z-00-DX1.6F4914E0-AB5D-4D5F-8BF6-FB862AA63A87',
                'TCGA-5P-A9KC-01Z-00-DX1.F3D67C35-111C-4EE6-A5F7-05CF8D01E783'
        ]:
            continue

        if os.path.isdir(
                f"patches/{args.cancer}/mag_{args.magnification}-size_{args.patch_size}/{row['id_svs']}"
        ):
            continue

        svs_fname = f"{row['id_svs']}.svs"
        print(svs_fname)
        # full_path = row['svs_path'].replace('/pool2','/home/sjiang')
        full_path = row['svs_path']
        svs_folder = full_path.replace(svs_fname, '')
        paired_inputs.append((svs_fname, svs_folder, args.cancer))

    _ = process_map(get_masks, paired_inputs, max_workers=args.num_workers)
    _ = process_map(get_patches, paired_inputs, max_workers=args.num_workers)
