import numpy as np


def __zero_mean(image):
    mean = np.mean(image, axis=(0, 1))
    # print('Mean: ', mean)
    return image - mean


def __unit_variance(image):
    std = np.std(image, axis=(0, 1))
    # print('Standard deviation: ', std)
    return image / std


def normalize(image):
    return __unit_variance(__zero_mean(image))
