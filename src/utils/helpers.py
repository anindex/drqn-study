from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import numpy as np
import cv2

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


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
    return np.mean(img, axis=2, keepdims=True)


def downsample(img):
    return img[::2, ::2]


def crop_atari(img):
    return img[10:]


def preprocessAtari(img):
    img = downsample(crop_atari(img))
    return img.reshape((-1, img.shape[0], img.shape[1]))


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


def plot_grad_flow(named_parameters):
    '''
    Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow
    '''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])


def plot_lstm_grad_over_steps(x, grad_mean_ih, grad_mean_hh, grad_max_ih, grad_max_hh):
    plt.subplot(2, 2, 1)
    plt.plot(x, grad_mean_ih)
    plt.title('Mean LSTM grad magnitude ih')
    plt.xlabel('Steps')
    plt.subplot(2, 2, 2)
    plt.plot(x, grad_mean_hh)
    plt.title('Mean LSTM grad magnitude hh')
    plt.xlabel('Steps')
    plt.subplot(2, 2, 3)
    plt.plot(x, grad_max_ih)
    plt.title('Max LSTM grad magnitude ih')
    plt.xlabel('Steps')
    plt.subplot(2, 2, 4)
    plt.plot(x, grad_max_hh)
    plt.title('Max LSTM grad magnitude hh')
    plt.xlabel('Steps')
    plt.show()


def plot_max_abs_q(x, max_abs_q_log):
    # max abs q
    plt.grid(True)
    max_abs_q_log = np.array(max_abs_q_log)
    max_abs_q_log = max_abs_q_log.max(axis=0).squeeze()
    plt.title('Max Absolute Q over steps')
    plt.ylabel('Max abs(Q)')
    plt.xlabel('Steps')
    plt.plot(x, max_abs_q_log)
    plt.show()
    plt.title('Histogram of log scale max abs(Q)')
    plt.ylabel('Counts')
    plt.xlabel('max abs(Q)')
    plt.hist(max_abs_q_log)
    plt.show()


def plot_holistic_measure(x, y, title='', xlabel='', ylabel=''):
    # tderr
    y = np.array(y)
    y_mean, y_var = y.mean(axis=0), y.var(axis=0)
    plt.plot(x, y_mean)
    plt.fill_between(x, y_mean - y_var, y_mean + y_var)
    plt.title(title)
    plt.ylabel(xlabel)
    plt.xlabel(ylabel)
    plt.show()


def save_data(d, path):
    np.save(path, d)
