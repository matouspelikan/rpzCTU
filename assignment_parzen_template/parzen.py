#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

# Recommended imports:
from scipy.stats import norm
import scipy.optimize as opt
import copy
from bayes import *


def my_parzen(x, x_trn, h):
    """
    p = my_parzen(x, x_trn, h)

    Parzen window density estimation with normal kernel N(0, h^2).

    :param x:       vector of data points where the probability density functions
                    should be evaluated <1 x n> np array
    :param x_trn:   training data <1 x m> np array
    :param h:       kernel bandwidth (scalar)
    :return:        estimated p(x|k) evaluated at values given by x <1 x n> np array
    """

    raise NotImplementedError("You have to implement this function.")
    return p


def compute_Lh(itrn, itst, x, h):
    """
    Lh = compute_Lh(itrn, itst, x, h)

    Computes the average log-likelihood over training/test splits generated
    by crossval for a fixed kernel bandwidth h.

    :param itrn:    LIST of <1 x n> np arrays data splits (indices) generated by crossval()
    :param itst:    LIST of <1 x n> np arrays data splits (indices) generated by crossval()
    :param x:       input data (scalar)
    :param h:       kernel bandwidth (scalar)
    :return:        Lh - average log-likelihood over training/test splits (scalar)
    """

    raise NotImplementedError("You have to implement this function.")
    return Lh


def classify_bayes_parzen(x_test, xA, xC, pA, pC, h_bestA, h_bestC):
    """
    labels = classify_bayes_parzen(x_test, xA, xC, pA, pC, h_bestA, h_bestC)

    Classifies data using bayesian classifier with densities estimated by
    Parzen window estimator.

    :param x_test:  data (measurements) to be classified <1 x n> np array
    :param xA:      training data for Parzen window for class A <1 x n> np array
    :param xC:      training data for Parzen window for class C <1 x n> np array
    :param pA:      prior probabilities (scalar)
    :param pC:      prior probabilities (scalar)
    :param h_bestA: optimal values of the kernel bandwidth (scalar)
    :param h_bestC: optimal values of the kernel bandwidth (scalar)
    :return:        labels - classification labels for x_test <1 x n> np array
    """

    raise NotImplementedError("You have to implement this function.")
    return labels


################################################################################
#####                                                                      #####
#####             Below this line are already prepared methods             #####
#####                                                                      #####
################################################################################


def show_classification(test_images, labels, letters):
    """
    show_classification(test_images, labels, letters)

    create montages of images according to estimated labels

    :param test_images:     shape h x w x n
    :param labels:          shape 1 x n
    :param letters:         string with letters, e.g. 'CN'
    """
    def montage(images, colormap='gray'):
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
        imgs = test_images[:,:,labels[0]==i]
        subfig = plt.subplot(1,len(letters),i+1)
        montage(imgs)
        plt.title(letters[i])


def crossval(num_data, num_folds):
    """
    itrn, itst = crossval(num_data, num_folds)

    Partitions data for cross-validation.

    This function randomly partitions data into the training
    and testing parts. The number of partitioning is determined
    by the num_folds. If num_folds==1 then makes only one random
    partitioning of data into training and testing in ratio 50:50.

    :param num_data:    number of data (scalar, integer)
    :param num_folds:   number of folders (scalar, integer)
    :return:            itrn - LIST of training folds, itst - LIST of testing folds
                        itrn[i] indices of training data of i-th folder <1 x n> np array
                        itst[i] indices of testing data of i-th folder <1 x n> np array
    """
    if num_folds < 2:
        num_folds = 2

    inx = np.expand_dims(np.random.permutation(num_data), 0)

    itrn = []
    itst = []

    num_column = np.int32(np.ceil(np.float64(num_data) / num_folds))

    for idx in range(num_folds):
        tst_idx = range((idx * num_column), np.min([num_data, ((idx + 1) * num_column)]))
        trn_idx = [i for i in list(range(num_data)) if i not in tst_idx]
        itst.append(inx[:, tst_idx])
        itrn.append(inx[:, trn_idx])
    return itrn, itst