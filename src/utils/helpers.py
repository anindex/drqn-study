from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import numpy as np
import cv2


def loggerConfig(log_file, verbose=2):
    logger = logging.getLogger()
    formatter = logging.Formatter('[%(levelname)-8s] (%(processName)-11s) %(message)s')
    fileHandler = logging.FileHandler(log_file, 'w')
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    if verbose >= 2:
        logger.setLevel(logging.DEBUG)
    elif verbose >= 1:
        logger.setLevel(logging.INFO)
    else:
        # NOTE: we currently use this level to log to get rid of visdom's info printouts
        logger.setLevel(logging.WARNING)
    return logger


def to_grayscale(img):
    return np.mean(img, axis=2).astype(np.uint8)


def downsample(img):
    return img[::2, ::2]


def preprocessAtari(img):
    return to_grayscale(downsample(img))


def crop_scale(img):
    img = img[34:34 + 160, :160]
    img = cv2.resize(img, (80, 80))
    img = cv2.resize(img, (42, 42))
    img = img.mean(2)
    img = img.astype(np.float32)
    img *= (1. / 255.)
    return img


# NOTE: this only works with Gym scene format
def rgb2gray(rgb):
    gray_image = 0.2126 * rgb[..., 0]
    gray_image[:] += 0.0722 * rgb[..., 1]
    gray_image[:] += 0.7152 * rgb[..., 2]
    return gray_image


def rgb2y(rgb):
    y_image = 0.299 * rgb[..., 0]
    y_image[:] += 0.587 * rgb[..., 1]
    y_image[:] += 0.114 * rgb[..., 2]
    return y_image


def scale(image, hei_image, wid_image):
    return cv2.resize(image, (wid_image, hei_image),
                      interpolation=cv2.INTER_LINEAR)


def one_hot(n_classes, labels):
    one_hot_labels = np.zeros(labels.shape + (n_classes,))
    for c in range(n_classes):
        one_hot_labels[labels == c, c] = 1
    return one_hot_labels
