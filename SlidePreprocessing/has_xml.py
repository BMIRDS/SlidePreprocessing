from pathlib import Path

def has_xml(slidepath: Path, slide_extension: str, opts=None):
    """Return counterpart xml file path if exist. Otherwise None.

    Assumption:
        Xml and slide shares the same stem:
            .../[parent_path]/[stem].[ext]
            where ext is xml and svs (or tif)
    """
    if opts and opts.get('xml_root', None):
        candidate_filename = Path(slidepath.stem).with_suffix('.xml')
        candidate_xmlpath = Path(opts.get('xml_root')) / candidate_filename
    else:
        # xml file exists in the same dir
        candidate_xmlpath = Path(slidepath).parent / Path(slidepath.stem).with_suffix('.xml')
    if candidate_xmlpath.exists():
        return candidate_xmlpath
    else:
        return None

def has_xml_user(slidepath, slide_extension, opts=None):
    """Return counterpart xml file path if exist. Otherwise None.

    User define has_xml function.

    Update the following logic
        if xml file name is different from slide file name
    """
    xml_root = opts["xml_root"]
    xml_path = None
    xml_path_debug = list()
    for p in Path(xml_root).rglob('*.xml'):
        # if slidepath.stem.rstrip().startswith(p.stem.rstrip()):  # can cause error when two slides has partially the same name
        slide_stem = slidepath.stem.rstrip(' ').replace('-', '_')
        if p.stem.replace('-', '_').startswith(slide_stem) and not p.stem[len(slide_stem)].isdigit():
            xml_path = p
            xml_path_debug.append(p)
    if len(xml_path_debug) > 1:
        print(f"Warning: multiple candidate xmls: {xml_path_debug}")
    return xml_path
