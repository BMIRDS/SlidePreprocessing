from pathlib import Path
from multiprocessing import Pool
from collections import defaultdict
from PIL import Image
import numpy
Image.MAX_IMAGE_PIXELS = None

from annotation import load_xml, create_polygon
from has_xml import has_xml

"""
Takes folders that contain pairs of tissues and masks.
Outputs small patches extracted from those tissues
"""

# Modified from utils_dataset to use mask
def extract_patches(img: Image, img_mask: Image, window_size: int = 224,
                    overlap_factor: float = 1/3, tissue_threshold=0.7,
                    xmlpath: str = None):
    """
    
    """
    try:
        assert img.size == img_mask.size
    except AssertionError as e:
        print(f"img and img_mask must be the same size")
    nonoverlap_factor = 1 - overlap_factor
    x_max, y_max = img.size
    x_steps = int((x_max-window_size) / window_size / nonoverlap_factor)  # number of x starting points
    y_steps = int((y_max-window_size) / window_size / nonoverlap_factor)  # number of y starting points
    step_size = int(window_size * nonoverlap_factor)  # step size, same for x and y

    """Extract candidate patches
    """
    coords = list()
    for i in range(x_steps+1):
        for j in range(y_steps+1):
            x_start = i*step_size
            x_end = x_start + window_size
            y_start = j*step_size
            y_end = y_start + window_size
            # assert x_start >= 0; assert y_start >= 0; assert x_end <= x_max; assert y_end <= y_max
            if x_end > x_max or y_end > y_max:
                continue
            bbox = [x_start, y_start, x_end, y_end]
            patch = img.crop(bbox)
            patch_mask = img_mask.crop(bbox)
            if numpy.array(patch_mask).mean() > 255*tissue_threshold:
                # at least 70% (default) should be tissue in a patch
                coords.append(
                    (x_start, y_start))
    if len(coords) == 0:
        return [], []
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    min_x, max_x = min(xs), max(xs)+window_size
    min_y, max_y = min(ys), max(ys)+window_size
    patches = list()
    positions = list()
    classes = list()
    for c in coords:
        x_i = (c[0]-min_x)//step_size + 1
        y_i = (c[1]-min_y)//step_size + 1
        positions.append(tuple([x_i, y_i]))
        crop = img.crop([c[0], c[1], c[0]+window_size, c[1]+window_size])
        patches.append(crop)
        is_found = False
        if xmlpath is not None:
            annotations = load_xml(xmlpath)
            for anno in annotations:
                name = anno['tree'].get('Name')
                polygon = anno['polygon']
                patch_coords = [c,
                     [c[0], c[1]+window_size],
                     [c[0]+window_size, c[1]+window_size],
                     [c[0]+window_size, c[1]]]
                if not is_found and create_polygon(patch_coords).intersects(polygon):
                    classes.append(name)
                    is_found = True
        if not is_found:
            classes.append("NotAnnotated")
    return patches, positions, classes


def save_patches(slide_name: str, tissues, dst_path: Path,
                 extraction_config, extract_notannotated: bool = False):
    """Wrapper function for extract_patches,
    to extract patches from tissues on a slide.

    extraction_config: dict
        requires the following keys for extract_patches function:
            window_size (default: 224)
            overlap_factor (default: 1/3)
            tissue_threshold (default: 0.7)
    """
    slide_dst = dst_path / slide_name
    slide_dst.mkdir(parents=True, exist_ok=True)
    for p, p_mask, p_xml in tissues:
        img = Image.open(p)
        img_mask = Image.open(p_mask)
        patches, positions, annotations = extract_patches(
            img=img,
            img_mask=img_mask,
            window_size=extraction_config.get('window_size', 224),
            overlap_factor=extraction_config.get('overlap_factor', 1/3),
            tissue_threshold=extraction_config.get('tissue_threshold', 0.7),
            xmlpath=p_xml)
        for patch, pos, annotation in zip(patches, positions, annotations):
            """Recommend to customize following logic:
            Use annotation info to decide if a patch need to be saved or
            saved in specific folder 
            """
            if annotation == 'NotAnnotated' and not extract_notannotated:
                continue
            (slide_dst / annotation).mkdir(parents=True, exist_ok=True)
            patch_dst = slide_dst / annotation / (
                p.stem + f"_{pos[0]}_{pos[1]}" + p.suffix)
            patch.save(patch_dst)


if __name__ == '__main__':
    # CONFIG
    from config import load_config
    config = load_config()
    src = config.src_2
    dst = config.dst_2
    num_workers = config.num_workers
    use_progress_bar = config.use_progress_bar
    image_suffix = config.image_suffix
    opts = [{'xml_root': opt1} for opt1 in config.opt1]  # placeholder for possible extension
    window_size = config.window_size
    overlap_factor = config.overlap_factor
    tissue_threshold = config.tissue_threshold
    extraction_config = {
        'window_size': window_size,
        'overlap_factor': overlap_factor,
        'tissue_threshold': tissue_threshold,
    }
    extract_notannotated = config.extract_notannotated

    for sp, dp in zip(src, dst):
        dp = Path(dp)

        pairs = defaultdict(list)
        for p in Path(sp).rglob(f'*.{image_suffix}'):
            p_mask = p.with_name(
                f'{p.stem}_mask').with_suffix(f'.{image_suffix}')
            if p_mask.exists():
                slide_name = p.parent.name
                pairs[slide_name].append(
                    (p, p_mask, has_xml(p, image_suffix)))
        if not config.multiprocess:
            """Single thread for debugging
            """
            for slide_name, tissues in pairs.items():
                save_patches(
                    slide_name = slide_name,
                    tissues = tissues,
                    dst_path = dp,
                    extraction_config=extraction_config,
                    extract_notannotated=extract_notannotated,)
        else:
            """ Multi-threading
            """
            pool = Pool(num_workers)
            for slide_name, tissues in pairs.items():
                pool.apply_async(
                            save_patches,
                            args=[],
                            kwds={
                                'slide_name': slide_name,
                                'tissues': tissues,
                                'dst_path': dp,
                                'extraction_config': extraction_config,
                                'extract_notannotated': extract_notannotated,
                                }
                            )
            pool.close()
            pool.join()
