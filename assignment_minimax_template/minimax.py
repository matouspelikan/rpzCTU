#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from bayes import *
import scipy.optimize as opt
import copy


def risk_fix_q_discrete(D1, D2, D1_priors, q):
    """
    risks = risk_fix_q_discrete(D1, D2, D1_priors, q)

    Computes risk(s) for varying prior.

    :param D1:          discrete distributions, priors not needed, pXk(x|D1) given as a <1 × n> np array
    :param D2:          discrete distributions, priors not needed, pXk(x|D2) given as a <1 × n> np array
    :param D1_priors:   D1_priors <1 x n> np array of D1 priors
    :param q:           strategy - <1 × n> np array, values 0 or 1
    :return:            <1xn> np array of bayesian risk of the strategy q
                        with 0-1 cost function and varying priors D1_priors
    """

    W = np.array([[0, 1], [1, 0]])

    raise NotImplementedError("You have to implement this function.")
    risks = None
    return risks


def worst_risk_discrete(D1, D2, D1_priors):
    """
    worst_risks = worst_risk_discrete(D1, D2, D1_priors)

    Compute worst possible risks of a bayesian strategies.

    :param D1:          discrete distributions, pXk(x|D1) given as a <1 × n> np array
    :param D2:          discrete distributions, pXk(x|D2) given as a <1 × n> np array
    :param D1_priors:   D1_priors <1 x n> np array of D1 priors
    :return:            <1 x n> worst risk of bayesian strategies
                        for D1, D2 with different priors D1_priors

    Hint: for all D1_priors calculate bayesian strategy and corresponding maximal risk.
    """
    W = np.array([[0., 1.], [1., 0.]])
    worst_risks = np.zeros_like(D1_priors)
    raise NotImplementedError("You have to implement this function.")

    return worst_risks


def minmax_strategy_discrete(D1, D2):
    """
    q, worst_risk = minmax_strategy_discrete(D1, D2)

    Find minmax strategy.

    :param D1:  discrete distributions, pXk(x|D1) given as a <1 × n> np array
    :param D2:  discrete distributions, pXk(x|D2) given as a <1 × n> np array
    :return:    q - strategy, <1 x n> np array of 0 and 1 (see find_strategy_discrete)
                worst_risk - worst risk of the minimax strategy q
    """
    W = np.array([[0., 1.], [1., 0.]])
    raise NotImplementedError("You have to implement this function.")
    q = None
    worst_risk = None

    return q, worst_risk


def risk_fix_q_cont(D1, D2, D1_priors, q):
    """
    function risks = risk_fix_q_cont(D1, D2, D1_priors, q)

    Computes risk(s) for varying prior.

    :param D1:          parameters of the normal dist.
                        D1['Mean'], D1['Sigma']
    :param D2:          parameters of the normal dist.
                        D2['Mean'], D2['Sigma']
    :param D1_priors:   <1xn> np array of D1 priors
    :param q:           strategy
                        q['t1'] q['t2'] - two descision thresholds
                        q['decision'] - 3 decisions for intervals (-inf, t1>, (t1, t2>, (t2, inf) shape <1 x 3>
    :return:            <1xn> np array of bayesian risk of the strategy q with
                        varying prior D1_priors
    """
    risks = np.zeros_like(D1_priors)
    raise NotImplementedError("You have to implement this function.")

    return risks


def worst_risk_cont(D1, D2, D1_priors):
    """
    worst_risks = worst_risk_cont(D1, D2, D1_priors)

    Compute worst possible risks of a bayesian strategies.

    :param D1:          parameters of the normal dist.
                        D1['Mean'], D1['Sigma']
    :param D2:          parameters of the normal dist.
                        D2['Mean'], D2['Sigma']
    :param D1_priors:   <1xn> np array of D1 priors
    :return:            <1 x n> worst risk of bayesian strategies
                        for D1, D2 with different priors D1_priors

    Hint: for all D1_priors calculate bayesian strategy and
    corresponding maximal risk.
    """
    worst_risks = np.zeros_like(D1_priors)
    raise NotImplementedError("You have to implement this function.")

    return worst_risks


def minmax_strategy_cont(D1, D2):
    """
    q, risk = minmax_strategy_cont(D1, D2)

    Find minmax strategy.

    :param D1:          parameters of the normal dist.
                        D1['Mean'], D1['Sigma']
    :param D2:          parameters of the normal dist.
                        D2['Mean'], D2['Sigma']
    :return:            q - strategy
                        q['t1'] q['t2'] - two descision thresholds
                        q['decision'] - 3 decisions for intervals (-inf, t1>, (t1, t2>, (t2, inf) shape <1 x 3>
                        worst_risk - worst risk of the minimax strategy q
    """

    raise NotImplementedError("You have to implement this function.")
    q = None
    worst_risk = None

    return q, worst_risk


################################################################################
#####                                                                      #####
#####             Below this line are already prepared methods             #####
#####                                                                      #####
################################################################################


def create_test_set(images_test, letters):
    """
    images, labels = create_test_set(images_test, letters)

    Return subset of the <images_test> corresponding to <letters>

    :param images_test: dict of all test images
                        images_test.A, images_test.B, ...
    :param letters:     string with letters, e.g. 'CN'
    :return:            images - shape h x w x n
                        labels - shape 1 x n
    """

    images = None
    labels = None
    for idx in range(len(letters)):
        imgs = images_test[letters[idx]]
        cur_labels = idx * np.ones([1, imgs.shape[2]], np.int32)
        if images is None:
            images = copy.copy(imgs)
            labels = cur_labels
        else:
            images = np.concatenate([images, imgs], axis=2)
            labels = np.concatenate([labels, cur_labels], axis=1)
    return images, labels


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
