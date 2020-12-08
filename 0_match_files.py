import os
from pathlib import Path

from config import load_config

""" Scan through files and match slides and corresponding annotation files,
and report if could not find a corresponding annotation file for a slide.

Check all files raised by `Warning`, because this script cannot tell
if it missed a corresponding xml file or an xml file does not exist.
You can safely ignore the latter case. (Maybe you should talk to pathologists
if all the slides are supposed to be annotated.)

To resolve unmatching, make sure
1. annotation files exist
2. annotation files are in the same directory as slides
   or xml_root path correctly nagivates to a dir storing all the xmls.

Still no match? Then it's time to update has_xml_user function in has_xml.py
because the file name format of your annotation files is in irregular form.

"""

if __name__ == '__main__':

    """ Loading config """
    config = load_config()
    src = config.src_1
    slide_extension = config.slide_extension
    opts = [{'xml_root': opt1} for opt1 in config.opt1]
    skip_slide_wo_xml = config.skip_slide_wo_xml

    if config.use_userdefined_has_xml:
        # Use CUSTOM has_xml function.
        from has_xml import has_xml_user as has_xml
    else:
        # Use DEFAULT has_xml function
        from has_xml import has_xml

    for sp, op in zip(src, opts):
        print(f"\nSource: {sp}")
        entries = list(Path(sp).rglob(f"*.{slide_extension}"))
        xmls = [has_xml(e, slide_extension, op) for e in entries]
        for slidepath, xmlpath in zip(entries, xmls):
            if xmlpath is None:
                print(f"Warning: No match for {slidepath}")
                continue
            else:
                print(f"Info: Match {slidepath.name}, {xmlpath.name}")

