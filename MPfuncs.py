import numpy as np
from skimage import color, filters, morphology
from scipy import ndimage as ndi


def intensity_func(img_slice):
    norm_gray = np.array(rgb2gray(img_slice / 255.0))
    scaled_gray = norm_gray * 255.0
    scaled_gray.astype("uint8")
    return scaled_gray


def iso_func(img_slice):
    return filters.threshold_isodata(img_slice)


def li_func(img_slice):
    return filters.threshold_li(img_slice)


def otsu_func(img_slice):
    return filters.threshold_otsu(img_slice)


def yen_func(img_slice):
    return filters.threshold_yen(img_slice)


def edge_func(img_slice):
    return filters.sobel(img_slice)


def watershed_func(img_slice):
    edge_data = img_slice[0]
    byw_data = img_slice[1]
    thresh_val = img_slice[2]
    markers = np.zeros_like(byw_data)
    markers[byw_data < thresh_val] = 1
    markers[byw_data >= thresh_val] = 2
    segment_mask = morphology.watershed(edge_data, markers)
    segment_mask = ndi.binary_fill_holes(segment_mask - 1)
    return segment_mask


def lab_func(img_slice):
    return color.rgb2lab(img_slice)


def rgb2gray(img_slice):
    r, g, b = img_slice[:, :, 0], img_slice[:, :, 1], img_slice[:, :, 2]
    gray = r / 3.0 + g / 3.0 + b / 3.0
    return gray
