#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

# Recommended imports:
from scipy.stats import norm
import scipy.optimize as opt
from bayes import *
from minimax import *


def mle_normal(x):
    """
    mu, sigma = mle_normal(x)

    Computes maximum likelihood estimate of mean and sigma of a normal distribution.

    :param x:   input features <1 x n>
    :return:    mu - mean
                sigma - standard deviation
    """
    raise NotImplementedError("You have to implement this function.")
    return mu, sigma


def mle_variance(cardinality):
    """
    var_mean, var_sigma = mle_variance(cardinality)

    Estimates variance of estimated parameters of a normal distribution in 100 trials.

    :param cardinality: size of the generated dataset (e.g. 1000)
    :return:            var_mean - variance of the estimated means in 100 trials
                        var_sigma - variance of the estimated standard deviation in 100 trials
    """
    num_trials = 100

    raise NotImplementedError("You have to implement this function.")
    return var_mean, var_sigma


def estimate_prior(id_label, labelling):
    """
    function prior = estimate_prior(id_label, labelling)

    Estimates prior probability of a class.

    :param id_label:    id of the selected class
    :param labelling:   <1 x n> vector of label ids
    :return:            prior probability
    """

    raise NotImplementedError("You have to implement this function.")
    return prior


def loglikelihood_sigma(x, D, sigmas):
    """
    L, maximizer_sigma, maxL = loglikelihood_sigma(x, D, sigmas)

    Compute log likelihoods and maximum ll of a normal distribution with fixed mean and variable standard deviation.

    :param x:       input features <1 x n>
    :param D:       D['Mean'] the normal distribution mean
    :param sigmas:  <1 x m> vector of standard deviations
    :return:        L - <1 x m> vector of log likelihoods
                    maximizer_sigma - sigma for the maximal log likelihood
                    max_L - maximal log likelihood

    Hint: use opt.fminbound()
    """

    # try to implement this without loops

    raise NotImplementedError("You have to implement this function.")
    return L, maximizer_sigma, max_L


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