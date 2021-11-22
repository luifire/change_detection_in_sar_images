import numpy as np
import math

from Global_Info import *

totalMean = 0
totalVariance = 0  # you can average std_dev by summing all variances and then taking the sqrt (the Internet)
example_count = 0

MEAN_VH_0 = 9.437738587279922
STD_DEV_VH_0 = 2.4122262625519295

MEAN_VV_1 = 11.58194002045524
STD_DEV_VV_1 = 2.6876261150196217


def normalize_image(img):
    global MEAN_VV_1
    global STD_DEV_VV_1
    global MEAN_VH_0
    global STD_DEV_VH_0

    log = np.log2(img + 1)

    std_norm_vh = log[VH_CHANNEL, :, :]
    std_norm_vh = (std_norm_vh - MEAN_VH_0) / STD_DEV_VH_0

    std_norm_vv = log[VV_CHANNEL, :, :]
    std_norm_vv = (std_norm_vv - MEAN_VV_1) / STD_DEV_VV_1

    stack = np.stack([std_norm_vh, std_norm_vv])
    return np.float32(stack)


def print_mean_and_std_dev():
    print("Mean")
    print(str(totalMean / example_count))
    print("Std Dev")
    print(str(math.sqrt(totalVariance / example_count)))


def denormalize(img, mean, std_dev):

    def exp(val):
        if val < 0:
            return -val * val
        else:
            return val * val

    std_norm_matrix = np.vectorize(lambda x: x * std_dev + mean)
    mat = std_norm_matrix(img)

    exp_matrix = np.vectorize(exp)
    mat = exp_matrix(mat)

    return mat

""" Mean Test Fct
    from image_manipulation import *
    from NormalizeData import *
    for f1, f2 in file_pairs:
        print(f1)
        img = read_sar_image(f1)
        if img is None:
            continue
        normalize_image(img[0,:,:])
        print_mean_and_std_dev()
    exit(0)
    
    
    
        def log(val):
        if val is None:
            return (MEAN_VH_0 + MEAN_VV_1) / 2
        val = math.log2(val + 1)
        return val

    if img is None:
        return None, -1, -1
"""

"""
    global totalMean
    global example_count
    global totalVariance

    mean = np.mean(mat)
    std_dev = math.sqrt(np.var(mat))
    print('current mean: ' + str(mean) + "  current std_dev" + str(std_dev))
    totalMean += mean
    totalVariance += std_dev * std_dev
    example_count += 1

    #std_norm_matrix = np.vectorize(lambda x: (x - mean) / std_dev)
    #mat = std_norm_matrix(mat)

    return np.float32(mat), mean, std_dev
"""