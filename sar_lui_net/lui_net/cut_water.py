import tifffile as tiff
import os

from PointsAndRectangles import *
from Global_Info import *


def _cut_water(tif_file, new_rect):
    img = tiff.imread(tif_file)
    img = img[:, new_rect.top:new_rect.bottom, new_rect.left:new_rect.right]

    tiff.imwrite(tif_file, img)


def _cut_area_in_dir(dir, new_rect):
    lst = list()
    for root, dirs, files in os.walk(dir):
        for name in files:
            if name.endswith('.tif'):
                lst.append(name)

    for idx, name in enumerate(lst):
        print(str(idx) + ' / ' + str(len(lst)) + ' - ' + name)
        _cut_water(os.path.join(dir, name), new_rect)

if __name__ == '__main__':
    #_cut_area_in_dir(ALTERNATIVE_TIF_PATH + 'pitonfournaise/144', Rect(0, 0, 1921, 980))
    # total 4272 976
    _cut_area_in_dir(ALTERNATIVE_TIF_PATH + 'pitonfournaise/151', Rect(1505, 0, 1153, 976))
