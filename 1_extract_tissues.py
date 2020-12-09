import os
import numpy
from PIL import Image
from pathlib import Path
from multiprocessing import Pool

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(2**64)
Image.MAX_IMAGE_PIXELS = None

from openslide import OpenSlide
import cv2
import tissueloc

from annotation import (
    load_xml, create_polygon, create_tree, append_tree, overlay_annotation, prettify)
from microscope import compute_magnification

def extract_tissues(filepath: str, destdir: str, target_magnification: float,
                    image_suffix: str, xmlpath: str = None,
                    tissueloc_config: dict = dict()):
    """
    Args:
        filepath: str
            A source slide file path.
        destdir: str
            A directory to store extracted images and related files.

        target_magnification: float
            Target magnification. Typical values are 10, 20, or 40
        image_suffix: str
            image file type (extension) of extracted tissues
        xmlpath: str = None
            A corresponding XML file path (if any). This function also create
            a copy of XML file with updated coordinates in `target_magnification`
        tissueloc_config: dict
            provide a dictionary with following keys to override the default values.
                max_img_size (default: 2048*2)
                    Max height and width for the size of slide with selected level.
                    Proper pyramidal slide level for the size is automatically selected.
                    (small effect on running time: 8s+ w/ 4k vs 2k)
                smooth_sigma (default: 13)
                    Gaussian smoothing sigma.
                    Kernel size.
                thresh_val (default: 0.80)
                    Thresholding value.
                    A theshold for binary mask.
                min_tissue_size (default: 10000)
                    Minimum tissue area.
                    The number of pixels.
                For details, see:
                    https://github.com/PingjunChen/tissueloc/blob/master/tissueloc/locate_tissue.py

    Generated files has the following in file name:
        minx
        miny
        ,which are top-left coords of extracted image at level0.

    """
    cnts, d_factor = tissueloc.locate_tissue_cnts(
        filepath,
        max_img_size=tissueloc_config.get('max_img_size', 2048*2),
        smooth_sigma=tissueloc_config.get('smooth_sigma', 13),
        thresh_val=tissueloc_config.get('thresh_val', 0.80),
        min_tissue_size=tissueloc_config.get('min_tissue_size', 10000))

    slide = OpenSlide(filepath)

    dest_root_path = Path(destdir) / Path(filepath).stem
    dest_root_path.mkdir(parents=True, exist_ok=True)
    if len(cnts) == 0:
        print(f"No tissue found: Skipping {filepath}")
        return

    results = compute_magnification(slide, target_magnification)
    original_magnification = results.get('original_magnification')
    target_level = results.get('target_level')
    ds_from_target_level = results.get('donwsampling_factor')
    ds_at_target_level = int(slide.level_downsamples[target_level])
    ds_from_level0 = ds_at_target_level * ds_from_target_level

    for i, cnt in enumerate(cnts):
        cnt *= d_factor  # Scale back contours To original scale
        (maxx, maxy) = cnt.max(axis=0)[0]
        (minx, miny) = cnt.min(axis=0)[0]
        size_x = maxx - minx
        size_y = maxy - miny
        size_x_at_target_level = int(size_x / ds_at_target_level)
        size_y_at_target_level = int(size_y / ds_at_target_level)
        size_x_at_target_magnification = int(size_x_at_target_level / ds_from_target_level)
        size_y_at_target_magnification = int(size_y_at_target_level / ds_from_target_level)

        # Extract tissue region at target level then resize to target magnification
        dest_path = dest_root_path / f"{minx}_{miny}.{image_suffix}"
        img = slide.read_region(
            location=(minx, miny),
            size=(size_x_at_target_level, size_y_at_target_level),  # size at target level
            level=target_level)
        image = img.convert('RGB')
        image = image.resize(
            (size_x_at_target_magnification, size_y_at_target_magnification))
        image.save(dest_path)

        # Generate tissue mask
        mask = numpy.zeros_like(image)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # Translate contour
        cnt_t = cnt.astype(float)
        cnt_t[:, :, 0] -= minx
        cnt_t[:, :, 1] -= miny
        # Scale contour        
        cnt_t /= ds_from_level0
        cnt_t = cnt_t.astype(int)
        cv2.fillPoly(mask, [cnt_t], 255)
        dest_mask_path = dest_path.parent / f"{minx}_{miny}_mask.{image_suffix}"
        dest_mask_path = str(dest_mask_path.absolute())
        cv2.imwrite(dest_mask_path, mask)

        if xmlpath is not None:
            annotations = load_xml(xmlpath)
            tree = create_tree()
            is_updated = False
            for anno in annotations:
                polygon = anno['polygon']
                subtree = anno['tree']
                coords = [xy.tolist() for xy in cnt.squeeze()]
                if create_polygon(coords).intersects(polygon):  # this operation is done at level0
                    # convert tree to target level and rect
                    for coord in subtree.iter('Coordinate'):
                        # Downscale and crop annotation
                        x_transformed = (float(coord.attrib['X']) - minx)/ds_from_level0
                        y_transformed = (float(coord.attrib['Y']) - miny)/ds_from_level0
                        coord.attrib['X'] = str(x_transformed)
                        coord.attrib['Y'] = str(y_transformed)

                    # save xml file
                    tree = append_tree(tree, subtree)
                    is_updated = True
            if is_updated:
                tree = prettify(tree)
                new_xml_file = dest_root_path / f'{minx}_{miny}.xml'
                with open(new_xml_file, "w") as file_out:
                    file_out.write(tree)
                overlay_annotation(
                    image_path=dest_path,
                    xml_path=new_xml_file,
                    image_suffix=image_suffix)


if __name__ == '__main__':
    # LOADING CONFIG
    from config import load_config
    config = load_config()
    src = config.src_1
    dst = config.dst_1
    num_workers = config.num_workers
    target_magnification = config.target_magnification
    use_progress_bar = config.use_progress_bar
    slide_extension = config.slide_extension
    image_suffix = config.image_suffix
    if hasattr(config, 'tissueloc_config'):
        tissueloc_config = config.tissueloc_config
    else:
        tissueloc_config = dict()
    opts = [{'xml_root': opt1} for opt1 in config.opt1]  # placeholder for possible extension
    skip_slide_wo_xml = config.skip_slide_wo_xml
    if config.use_userdefined_has_xml:
        # Use CUSTOM has_xml function.
        from has_xml import has_xml_user as has_xml
    else:
        # Use DEFAULT has_xml function
        from has_xml import has_xml

    progress_bar_available = use_progress_bar
    if use_progress_bar:
        progress_bar_available = True
        try:
            from miniutils import parallel_progbar
            # dependency: nose

        except ImportError as e:
            print("Progress bar not available: {e}")
            progress_bar_available = False

    """Single threading for debugging"""   
    if not config.multiprocess:
        for sp, op, dp in zip(src, opts, dst):
            entries = list(Path(sp).rglob(f"*.{slide_extension}"))
            xmls = [has_xml(e, slide_extension, op) for e in entries]
            for slidepath, xmlpath in zip(entries, xmls):
                if xmlpath is None:
                    if skip_slide_wo_xml:
                        continue
                    else:
                        raise Exception(f"No corresponding XML file for {slidepath}. "
                            "If this is expected, set skip_slide_wo_xml to True."
                            "Otherwise check file name and update has_xml_user"
                            "function to locate the XML file.") 
                print("Processing ", slidepath, xmlpath)
                extract_tissues(
                        filepath=str(slidepath.absolute()),
                        destdir=dp,
                        target_magnification=target_magnification,
                        image_suffix=image_suffix,
                        xmlpath=xmlpath,
                        tissueloc_config=tissueloc_config)
        exit()

    """ Multi-threading"""
    for sp, op, dp in zip(src, opts, dst):
        if progress_bar_available:
            entries = list(Path(sp).rglob(f"*.{slide_extension}"))
            xmls = [has_xml(e, slide_extension, op) for e in entries]
            if not all(xmls):
                if skip_slide_wo_xml:
                    # Removing slides with no xml file.
                    entries_xmls = [(ex) for ex in zip(entries, xmls) if ex[1]]
                    entries, xmls = zip(*entries_xmls)
                else:
                    raise Exception(f"No corresponding XML file for {slidepath}. "
                        "If this is expected, set skip_slide_wo_xml to True."
                        "Otherwise check file name and update has_xml_user"
                        "function to locate the XML file.") 

            parallel_progbar(mapper=lambda x: extract_tissues(
                filepath=str(x[0].absolute()),
                destdir=dp,
                target_magnification=target_magnification,
                image_suffix=image_suffix,
                xmlpath=x[1],
                tissueloc_config=tissueloc_config,
                ), iterable=zip(entries, xmls),
                nprocs=num_workers, total=len(entries))
        else:
            pool = Pool(num_workers)
            for e in Path(sp).rglob(f"*.{slide_extension}"):
                xmlpath = has_xml(e, slide_extension, op)
                if xmlpath is None:
                    if skip_slide_wo_xml:
                        continue
                    else:
                        raise Exception(f"No corresponding XML file for {e}. "
                            "If this is expected, set skip_slide_wo_xml to True."
                            "Otherwise check file name and update has_xml_user"
                            "function to locate the XML file.") 
                if xmlpath is not None:
                    xmlpath = str(xmlpath.absolute())
                pool.apply_async(
                    extract_tissues,
                    args=[],
                    kwds={
                        'filepath': str(e.absolute()),
                        'destdir': dp,
                        'target_magnification': target_magnification,
                        'image_suffix': image_suffix,
                        'xmlpath': xmlpath,
                        'tissueloc_config': tissueloc_config,
                        }
                    )
            pool.close()
            pool.join()
