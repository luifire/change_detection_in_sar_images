import warnings
import tifffile as tiff
import numpy as np
from Global_Info import *


def make_images_same_size(img_1, img_2):

    def _make_1_same_size_as_2(own, other):
        # we are only making pictures smaller
        # when own_width > other_width then other_width else own_width
        new_1 = other.shape[1] if own.shape[1] > other.shape[1] else own.shape[1]
        new_2 = other.shape[2] if own.shape[2] > other.shape[2] else own.shape[2]

        new_img = own[:, 0:new_1, 0:new_2]
        return new_img

    return _make_1_same_size_as_2(img_1, img_2), _make_1_same_size_as_2(img_2, img_1)


def read_sar_image(image_path):
    img = tiff.imread(image_path)

    if img.shape[0] <= DIMENSION_OF_ORIGINAL_IMAGE or img.shape[0] != 4:
        warnings.warn("Picture only has " + str(img.shape[0]) + " dimensions " + str(img.shape) + image_path)
        return None

    def compute_magnitude(dim_1, dim_2):
        img_1 = img[dim_1, :, :]
        img_2 = img[dim_2, :, :]

        img_1 = np.square(img_1)
        img_2 = np.square(img_2)

        return img_1 + img_2

    vh = compute_magnitude(0, 1)
    vv = compute_magnitude(2, 3)

    return np.stack((vh, vv))


def make_image_3_channeled(img):
    if img.ndim != 3:
        return np.stack((img, img, img), axis=-1)

    return img


def combine_dimensions_of_predicted_img(img):
    # as eventually we will sum up all values squared,
    # looking at the quared average seems most natural to me
    quad = img ** 2
    diff = (quad[:, :, 0] + quad[:, :, 1]) / 2
    return diff
