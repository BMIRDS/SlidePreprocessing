from pathlib import Path
import argparse
import csv
import imghdr

from openslide import OpenSlide

"""Script to extract meta info from slides (svs, tiff)
"""

# Params
parser = argparse.ArgumentParser(
    description="Script to extract meta info from slides.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--input', '-i',
    type=str,
    required=True,
    help="Set a path to a directory that contains slides.\n"
    "ex) /pool2/data/WSI_EC")
parser.add_argument(
    '--out_filename', '-o',
    type=str,
    required=True,
    help="Set a CSV file name for outputting collected data.\n"
    "ex) meta_WSI_EC.csv")
parser.add_argument(
    '-x',
    type=str,
    default='svs',
    help="target file extension")
parser.add_argument(
    '--is_tcga',
    action='store_true',
    default=False,
    help="Use this flag for TCGA data to add extra column, "
    "slide-type indicating if a slide is frozen or FFPE.")
args = parser.parse_args()
target_dir = args.input
out_filename = args.out_filename
is_tcga = args.is_tcga


class TCGAUtils:
    ID_FROZEN = ['TS', 'BS']
    ID_PARAFFIN = ['DX']

    def _extract_type(self, s):
        return s.name.split('.')[0].split('-')[-1]

    def _is_type(self, t, IDs):
        return any([i in t for i in IDs])

    def detect_slidetype(self, file: Path):
        type_raw = self._extract_type(file)
        t = 'Unknown'
        if self._is_type(type_raw, self.ID_FROZEN):
            t = "Frozen"
        elif self._is_type(type_raw, self.ID_PARAFFIN):
            t = "Paraffin"
        return t


csvfile = open(out_filename, 'w', newline='')
writer = csv.writer(csvfile, delimiter=',')
files = Path(target_dir).rglob(f'*.{args.x}')
fields = list()
items = list()
for file in files:
    s = OpenSlide(str(file.absolute()))
    props = dict(s.properties)
    fields += props.keys()
    items.append([file, props])
fields = ['file'] + list(set(fields))  # largest key set
if is_tcga:
    fields.append('slide-type')
    tcga_utils = TCGAUtils()
writer.writerow(fields)
for file, props in items:
    rowdata = [file] + [props[k] if k in props else "" for k in fields[1:]]
    if is_tcga:
        t = tcga_utils.detect_slidetype(file)
        rowdata[-1] = t
    writer.writerow(rowdata)
csvfile.close()
