#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from PIL import Image

import scipy.optimize as opt
import copy
# importing bayes doesn't work in BRUTE :(, please copy the functions into this file


def risk_fix_q_discrete(discrete_A, discrete_B, discrete_A_priors, q):
    """
    Computes risk(s) for varying priors and 0-1 loss.

    :param discrete_A['Prob']:      pXk(x|A) given as a (n, ) np.array
    :param discrete_B['Prob']:      pXk(x|B) given as a (n, ) np.array
    :param discrete_A_priors:       discrete_A_priors (m, ) np.array
    :param q:                       strategy - (n, ) np.array, values 0 or 1 (see find_strategy_discrete)
    :return risks:                  bayesian risk of the strategy q (m, ) np.array
    """
    raise NotImplementedError("You have to implement this function.")
    risks = None
    return risks


def worst_risk_discrete(discrete_A, discrete_B, discrete_A_priors):
    """
    For each given prior probability value of the first class, the function finds the optimal bayesian strategy and computes its worst possible risk in case it is run with data from different a priori probability. It assumes the 0-1 loss.

    :param discrete_A['Prob']:      pXk(x|A) given as a (n, ) np.array
    :param discrete_B['Prob']:      pXk(x|B) given as a (n, ) np.array
    :param discrete_A_priors:       discrete_A_priors (m, ) np.array
    :return worst_risks:            worst risk of bayesian strategies (m, ) np.array
                                    for discrete_A, discrete_B with different discrete_A_priors

    Hint: for all discrete_A_priors calculate bayesian strategy and corresponding maximal possible risk if the real prior would be different.
    """
    raise NotImplementedError("You have to implement this function.")
    worst_risks = None
    return worst_risks


def minmax_strategy_discrete(discrete_A, discrete_B):
    """
    Find minmax strategy with 0-1 loss.

    :param discrete_A['Prob']:      pXk(x|A) given as a (n, ) np.array
    :param discrete_B['Prob']:      pXk(x|B) given as a (n, ) np.array
    :return q:                      strategy - (n, ) np.array, values 0 or 1 (see find_strategy_discrete)
    :return worst_risk              worst risk of the minimax strategy q, python float
    """
    raise NotImplementedError("You have to implement this function.")
    q, worst_risk = None, None
    return q, worst_risk


def risk_fix_q_cont(distribution_A, distribution_B, distribution_A_priors, q):
    """
    Computes bayesian risks for fixed strategy and various priors.

    :param distribution_A:          parameters of the normal dist.
                                    distribution_A['Mean'], distribution_A['Sigma'] - python floats
    :param distribution_B:          the same as distribution_A
    :param distribution_A_priors:   priors (n, ) np.array
    :param q:                       strategy dict - see bayes.find_strategy_2normal
                                       q['t1'], q['t2'] - decision thresholds - python floats
                                       q['decision'] - (3, ) np.int32 np.array decisions for intervals (-inf, t1>, (t1, t2>, (t2, inf)
    :return risks:                  bayesian risk of the strategy q with varying priors (n, ) np.array
    """
    raise NotImplementedError("You have to implement this function.")
    risks = None
    return risks


def worst_risk_cont(distribution_A, distribution_B, distribution_A_priors):
    """
    For each given prior probability value of the first class, the function finds the optimal bayesian strategy and computes its worst possible risk in case it is run with data from different a priori probability. It assumes the 0-1 loss.

    :param distribution_A:          parameters of the normal dist.
                                    distribution_A['Mean'], distribution_A['Sigma'] - python floats
    :param distribution_B:          the same as distribution_A
    :param distribution_A_priors:   priors (n, ) np.array
    :return worst_risk:             worst bayesian risk with varying priors (n, ) np.array
                                    for distribution_A, distribution_B with different priors

    Hint: for all distribution_A_priors calculate bayesian strategy and corresponding maximal risk.
    """
    raise NotImplementedError("You have to implement this function.")
    worst_risks = None
    return worst_risks


def minmax_strategy_cont(distribution_A, distribution_B):
    """
    q, worst_risk = minmax_strategy_cont(distribution_A, distribution_B)

    Find minmax strategy.

    :param distribution_A:  parameters of the normal dist.
                            distribution_A['Mean'], distribution_A['Sigma'] - python floats
    :param distribution_B:  the same as distribution_A
    :return q:              strategy dict - see bayes.find_strategy_2normal
                               q['t1'], q['t2'] - decision thresholds - python floats
                               q['decision'] - (3, ) np.int32 np.array decisions for intervals (-inf, t1>, (t1, t2>, (t2, inf)
    :return worst_risk      worst risk of the minimax strategy q - python float
    """
    raise NotImplementedError("You have to implement this function.")
    q, worst_risk = None, None
    return q, worst_risk


###########################################################################################
#  Put functions from previous labs here. (Sorry, we know imports would be much better)   #
###########################################################################################


################################################################################
#####                                                                      #####
#####             Below this line are already prepared methods             #####
#####                                                                      #####
################################################################################


def compute_measurement_lr_cont(imgs):
    """
    x = compute_measurement_lr_cont(imgs)

    Compute measurement on images, subtract sum of right half from sum of
    left half.

    :param imgs:    set of images, (h, w, n)
    :return x:      measurements, (n, )
    """
    assert len(imgs.shape) == 3

    width = imgs.shape[1]
    sum_rows = np.sum(imgs, dtype=np.float64, axis=0)

    x = np.sum(sum_rows[0:int(width / 2),:], axis=0) - np.sum(sum_rows[int(width / 2):,:], axis=0)

    assert x.shape == (imgs.shape[2], )
    return x


def compute_measurement_lr_discrete(imgs):
    """
    x = compute_measurement_lr_discrete(imgs)

    Calculates difference between left and right half of image(s).

    :param imgs:    set of images, (h, w, n) (or for color images (h, w, 3, n)) np.array
    :return x:      measurements, (n, ) np.array of values in range <-10, 10>,
    """
    assert len(imgs.shape) in (3, 4)
    assert (imgs.shape[2] == 3 or len(imgs.shape) == 3)

    mu = -563.9
    sigma = 2001.6

    if len(imgs.shape) == 3:
        imgs = np.expand_dims(imgs, axis=2)

    imgs = imgs.astype(np.int32)
    height, width, channels, count = imgs.shape

    x_raw = np.sum(np.sum(np.sum(imgs[:, 0:int(width / 2), :, :], axis=0), axis=0), axis=0) - \
            np.sum(np.sum(np.sum(imgs[:, int(width / 2):, :, :], axis=0), axis=0), axis=0)
    x_raw = np.squeeze(x_raw)

    x = np.atleast_1d(np.round((x_raw - mu) / (2 * sigma) * 10))
    x[x > 10] = 10
    x[x < -10] = -10

    assert x.shape == (imgs.shape[-1], )
    return x


def create_test_set(images_test, letters):
    """
    images, labels = create_test_set(images_test, letters)

    Return subset of the <images_test> corresponding to <letters>

    :param images_test: dict of all test images
                        images_test.A, images_test.B, ...
    :param letters:     string with letters, e.g. 'CN'
    :return images:     images - np.array (h, w, n)
    :return labels:     labels for images np.array (n,)
    """

    images = None
    labels = None
    for idx in range(len(letters)):
        imgs = images_test[letters[idx]]
        cur_labels = idx * np.ones([imgs.shape[2]], np.int32)
        if images is None:
            images = copy.copy(imgs)
            labels = cur_labels
        else:
            images = np.concatenate([images, imgs], axis=2)
            labels = np.concatenate([labels, cur_labels], axis=0)
    return images, labels


def show_classification(test_images, labels, letters):
    """
    show_classification(test_images, labels, letters)

    create montages of images according to estimated labels

    :param test_images:     np.array (h, w, n)
    :param labels:          labels for input images np.array (n,)
    :param letters:         string with letters, e.g. 'CN'
    """

    def montage(images, colormap='gray'):
        """
        Show images in grid.

        :param images:      np.array (h, w, n)
        :param colormap:    numpy colormap
        """
        h, w, count = np.shape(images)
        h_sq = np.int(np.ceil(np.sqrt(count)))
        w_sq = h_sq
        im_matrix = np.zeros((h_sq * h, w_sq * w))

        image_id = 0
        for j in range(h_sq):
            for k in range(w_sq):
                if image_id >= count:
                    break
                slice_w = j * h
                slice_h = k * w
                im_matrix[slice_h:slice_h + w, slice_w:slice_w + h] = images[:, :, image_id]
                image_id += 1
        plt.imshow(im_matrix, cmap=colormap)
        plt.axis('off')
        return im_matrix

    for i in range(len(letters)):
        imgs = test_images[:,:,labels==i]
        subfig = plt.subplot(1,len(letters),i+1)
        montage(imgs)
        plt.title(letters[i])
