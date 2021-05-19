import csv
import os
import numpy
import traceback
from PIL import Image
from pathlib import Path
from multiprocessing import Pool

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(2**63)
Image.MAX_IMAGE_PIXELS = None

from openslide import OpenSlide
import cv2
import tissueloc
import openslide

from annotation import (
    load_xml, create_polygon, create_tree, append_tree, overlay_annotation, prettify)
from microscope import compute_magnification

def extract_tissues(filepath: str, destdir: str, target_magnification: float,
                    image_suffix: str, xmlpath: str = None,
                    tissueloc_config: dict = dict(),
                    flattening = False, nickname=None):
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
        flattening: bool
            generate all the images in a single folder rather than creating a sub folder for each slide image.
        nickname: str
            custom name for the input slide, which can be used for the destination directory
            name. If this is not set, the stem of the slide is used.
    Generated files has the following in file name:
        minx
        miny
        ,which are top-left coords of extracted image at level0.

    As a biproduct, a csv file is also generated to store the details of
    extracted tissues. The csv file is created where the tissue images are stored.
        CSV: <slidename>.csv
        tissue_filename | original_left | original_top | original_right
         | original_bottom | original_magnification | donwsampling_factor

    """
    print(f"[info] Processing slide: {filepath}, xml: {xmlpath}")
    try:
        cnts, d_factor = tissueloc.locate_tissue_cnts(
            filepath,
            max_img_size=tissueloc_config.get('max_img_size', 2048*2),
            smooth_sigma=tissueloc_config.get('smooth_sigma', 13),
            thresh_val=tissueloc_config.get('thresh_val', 0.80),
            min_tissue_size=tissueloc_config.get('min_tissue_size', 10000))
    except openslide.lowlevel.OpenSlideError as e:
        # print(traceback.print_exc())
        print(f"[error] Extraction failed and thus skipped: {filepath} is not be compatible with openslide.",
            "(Recommendation: ignore this slide; such files are not multi-level and/or do not have objective-power value).")
        return False
    slide = OpenSlide(filepath)

    if nickname:
        stem = nickname
    else:
        stem = Path(filepath).stem

    if flattening:
        dest_root_path = Path(destdir)
        dest_root_path.mkdir(parents=True, exist_ok=True)

        dest_path_template = dest_root_path / (
            stem + "_{minx}_{miny}.{image_suffix}")
        dest_mask_path_template = dest_root_path / (
            stem + "_{minx}_{miny}_mask.{image_suffix}")
        new_xml_file_template = dest_root_path / (
            stem + '_{minx}_{miny}.xml')
    else:
        dest_root_path = Path(destdir) / stem
        dest_root_path.mkdir(parents=True, exist_ok=True)

        dest_path_template = dest_root_path / "{minx}_{miny}.{image_suffix}"
        dest_mask_path_template = dest_root_path / "{minx}_{miny}_mask.{image_suffix}"
        new_xml_file_template = dest_root_path / '{minx}_{miny}.xml'
    if len(cnts) == 0:
        print(f"[error] No tissue found: Skipping {filepath}. "
            "Reduce min_tissue_size parameter may help locating small tissues.")
        return False

    results = compute_magnification(slide, target_magnification)
    original_magnification = results.get('original_magnification')
    target_level = results.get('target_level')
    ds_from_target_level = results.get('donwsampling_factor')
    ds_at_target_level = int(slide.level_downsamples[target_level])
    ds_from_level0 = ds_at_target_level * ds_from_target_level

    rows = list()  # cache tissue info
    rows.append([
        'tissue_filename', 'original_left', 'original_top', 'original_right',
        'original_bottom', 'original_magnification', 'donwsampling_factor'])

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
        dest_path = str(dest_path_template).format(
            minx=minx, miny=miny, image_suffix=image_suffix)
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
        dest_mask_path = str(dest_mask_path_template.absolute()).format(
            minx=minx, miny=miny, image_suffix=image_suffix)
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
                new_xml_file = str(new_xml_file_template).format(
                    minx=minx, miny=miny)
                with open(new_xml_file, "w") as file_out:
                    file_out.write(tree)
                overlay_annotation(
                    image_path=dest_path,
                    xml_path=new_xml_file,
                    image_suffix=image_suffix)
        rows.append([
            str(dest_path), str(minx), str(miny), str(maxx), str(maxy),
            str(original_magnification), str(ds_from_level0)])

    with open(dest_root_path/f"{stem}.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

def path_parser(src, opts, dst, csv_as_data_source=False):
    """
    Return:
        a list of [(slideid, slidepath, xmlpath, destdir), ]
        which is a nested list; outer list is for multiple datasource, and
        inner list is for each slide.
    """
    results = list()
    for sp, op, dp in zip(src, opts, dst):
        paths = list()

        if csv_as_data_source:
            with open(sp, newline='') as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                for i, row in enumerate(reader):
                    if len(row) == 2:
                        # no xml
                        slideid, slidepath = row
                        xmlpath = None
                    elif len(row) == 3:
                        # has xml path
                        slideid, slidepath, xmlpath = row
                    else:
                        raise Exception(f"[error] The source CSV file {sp} is il-formed. "
                            "Please make sure if the file have either two or three columns.")
                    if not slideid:
                        # slideid cell is empty, so use file name as id
                        slideid = Path(slidepath).stem
                    paths.append((slideid, slidepath, xmlpath, dp))

        else:
            entries = list(Path(sp).rglob(f"*.{slide_extension}"))
            xmls = [has_xml(e, slide_extension, op) for e in entries]
            for slidepath, xmlpath in zip(entries, xmls):
                if xmlpath is None:
                    if action_no_xml == 'skip':
                        print(f"[info] Skipping slide: {slidepath} for no xml (see \'action_no_xml\'')")
                        continue
                    elif action_no_xml == 'raise':
                        raise Exception(f"[error] No corresponding XML file for \'{slidepath}\'. "
                            "If this is expected, set \'action_no_xml\' to \'skip\' or \'process\'' in a config file. "
                            "Otherwise, check the filename of an XML file and "
                            "update \'has_xml_user\' function to locate the XML file. "
                            "Running \'0_match_files.py\' helps to tell if "
                            "the slide-xml pair is located.") 
                slideid = path(slidepath).stem
                paths.append((slideid, slidepath, xmlpath, dp))
        results.append(paths)
    return results


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
    flattening = config.flattening
    if hasattr(config, 'tissueloc_config'):
        tissueloc_config = config.tissueloc_config
    else:
        tissueloc_config = dict()

    csv_as_data_source = hasattr(config, 'csv_as_data_source') and config.csv_as_data_source

    assert len(src) == len(dst),\
        "[error] The number of source and destination folders does not match. "\
        "Double check \"src_1\" and \"dst_1\" parameters in the config file."
    
    opts = [{'xml_root': opt1} for opt1 in config.opt1]  # placeholder for possible extension
    if len(src) != len(opts):
        # Validating parameters
        if len(opts) == 1 and not opts[0]['xml_root']:
            # Auto-expanding a trivial option parameter
            opts = opts*len(src)
        else:
            # Non trivial "xml_root" is set, and thus auto-expanding is risky so
            # raising an error for manual fix
            assert len(src) == len(opts),\
            "[error] The number of source folders and options does not match. "\
            "Double check \"src_1\" and \"opt1\" parameters in the config file."

    action_no_xml = config.action_no_xml
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
            print(f"[warning] Progress bar not available: {e}")
            progress_bar_available = False


    paths_sources = path_parser(src, opts, dst, csv_as_data_source=csv_as_data_source)

    """Single Process Mode for debugging"""
    if not config.multiprocess:
        print("[info] Mode: Single Process")
        for paths in paths_sources:  # each data source
            for slideid, slidepath, xmlpath, destdir in paths:
                extract_tissues(
                        filepath=str(Path(slidepath).absolute()),
                        destdir=destdir,
                        target_magnification=target_magnification,
                        image_suffix=image_suffix,
                        xmlpath=xmlpath,
                        tissueloc_config=tissueloc_config,
                        flattening=flattening,
                        nickname=slideid)

        exit()

    """ Multi-Process"""
    for paths in paths_sources:  # each data source
        print(f"[info] Mode: Multi-Process x{num_workers}")
        if progress_bar_available:
            parallel_progbar(
                mapper=lambda x: extract_tissues(
                    filepath=str(Path(x[1]).absolute()),
                    destdir=x[3],
                    target_magnification=target_magnification,
                    image_suffix=image_suffix,
                    xmlpath=x[2],
                    tissueloc_config=tissueloc_config,
                    flattening=flattening,
                    nickname=x[0]
                    ),
                    iterable=paths,
                nprocs=num_workers, total=len(paths))


        else:
            pool = Pool(num_workers)
            results = []
            for slideid, slidepath, xmlpath, destdir in paths:
                x = pool.apply_async(
                    extract_tissues,
                    args=[],
                    kwds={
                        'filepath': str(Path(slidepath).absolute()),
                        'destdir': destdir,
                        'target_magnification': target_magnification,
                        'image_suffix': image_suffix,
                        'xmlpath': xmlpath,
                        'tissueloc_config': tissueloc_config,
                        'flattening': flattening,
                        'nickname': slideid
                        }
                    )
                results.append(x)

            pool.close()
            for result in results:
                result.get()
            pool.join()
