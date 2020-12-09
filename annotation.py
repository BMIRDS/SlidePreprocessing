
import xml.etree.ElementTree as ET
from xml.dom import minidom
from shapely.geometry import Polygon  # conda install shapely
from pathlib import Path
from shutil import copy2
from collections import defaultdict
from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None

"""
Functions to load XML annotation files generated with ASAP for slides.
"""


def load_xml(xml_path):
    """Returns
        list of annotation polygons
        tree object
    """
    root = ET.parse(xml_path).getroot()
    annotations = list()
    for anno in root.iter('Annotation'):
        name = anno.get('PartOfGroup')
        coords = [(float(coord.get('X')), float(coord.get('Y')))
                  for coord in anno.iter('Coordinate')]
        try:
            polygon = Polygon(coords)
        except ValueError as e:
            print(e, coords)
            continue
        if not polygon.is_valid:
            polygon = polygon.buffer(0)  # fix self-interaction
        annotations.append(
            {'polygon': polygon,
             'tree': anno})
    return annotations

def create_polygon(coords):
    return Polygon(coords)


def prettify(tree):
    rough_string = ET.tostring(tree.getroot(), 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

def append_tree(tree, subtree):
    tree.find('Annotations').append(subtree)
    return tree


def create_tree(tag='ASAP_Annotations'):
    root = ET.Element(tag)
    tree = ET.ElementTree(root)
    node = ET.SubElement(tree.getroot(), 'Annotations')
    # node.append(subtree)
    return tree


def scale_level2_patch_to_original(topleft, patch_width, patch_height):
    """
    downsampling rate: 2
    This only works on my pipeline
    """
    ds_rate = 2
    patch_level = 2
    scale_factor = ds_rate**patch_level
    coords = [
        topleft,
        (topleft[0], topleft[1] + patch_height * scale_factor),
        (topleft[0] + patch_width * scale_factor, topleft[1] + patch_height * scale_factor),
        (topleft[0] + patch_width * scale_factor, topleft[1]),
    ]
    roi_polygon = Polygon(coords)
    return roi_polygon


def compute_interaction(roi_polygon, annotations):
    """
    topleft, width, height params are in level0
    annotations: a dictionary with includes, excludes keys
        Feed output of load_xml.
    """
    # coords = [
    #     topleft,
    #     (topleft[0], topleft[1] + height * 4),
    #     (topleft[0] + width * 4, topleft[1] + height * 4),
    #     (topleft[0] + width * 4, topleft[1]),
    # ]
    in_area = 0
    ex_area = 0

    for in_anno in annotations['includes']:
        in_area += in_anno.intersection(roi_polygon).area
    for ex_anno in annotations['excludes']:
        ex_area += ex_anno.intersection(roi_polygon).area
    # print(f"debug: in {in_area}, ex {ex_area}")
    return in_area - ex_area


"""Visual Validation
"""
# for stage 1
def overlay_annotation(image_path, xml_path, color='green', image_suffix='png'):
    """Overlay annotation on tissue image
    Args:
        image_path: Regular image (not SVS/TIF)
        xml_path
        color
        image_suffix
    """
    annotations = load_xml(xml_path)
    img = Image.open(image_path)
    img2 = img.copy()
    draw = ImageDraw.Draw(img2)
    # if not hasattr(annotations[0]['polygon'], "exterior"):
    #     print(image_path)
    #     print(annotations[0])
    #     print(annotations)
    draw.polygon(annotations[0]['polygon'].exterior.coords, fill=color)
    img3 = Image.blend(img, img2, 0.5)
    overlap_path = Path(image_path)
    overlap_path = overlap_path.parent / (overlap_path.stem + f'_overlay.{image_suffix}')
    img3.save(overlap_path)


def print_annotation_stats(xml_root):
    """
    Args:
        xml_root: str
            root path of all xml files
    """
    stats = defaultdict(int)
    for p in Path(xml_root).rglob('*.xml'):
        annotations = load_xml(p)
        for anno in annotations:
            name = anno['tree'].get('Name')
            polygon = anno['polygon']
            stats[name] += int(polygon.area)
            print(f"Slide: {p}\nClass: {name} Area: {int(polygon.area)}\n")
