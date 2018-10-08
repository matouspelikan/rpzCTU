#!/usr/bin/python
# -*- coding: utf-8 -*-

from scipy.stats import norm
import math
import numpy as np
import scipy.io
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
from scipy.stats import norm
from PIL import Image


def unwrap(data):
    try:
        while (len(data) == 1) and (len(data.shape) > 0):
            data = data[0]
    except TypeError:
        pass
    except:
        pass
    return data


def bayes_risk_discrete(discrete_A, discrete_B, W, q):
    """
    R = bayes_risk_discrete(discrete_A, discrete_B, W, q)

    Compute bayesian risk for a discrete strategy q

    :param discrete_A['Prob']:      pXk(x|A) given as a <1 × n> np array
    :param discrete_A['Prior']:     prior probability pK(A)
    :param discrete_B['Prob']:      pXk(x|B) given as a <1 × n> np array
    :param discrete_B['Prior']:     prior probability pK(B)
    :param W:                       cost function matrix
                                    dims: <states x decisions>
                                    (nr. of states and decisions is fixed to 2)
    :param q:                       strategy - <1 × n> np array, values 0 or 1
    :return:                        bayesian risk, scalar
    """

    raise NotImplementedError("You have to implement this function.")
    R = None
    return R


def find_strategy_discrete(discrete_A, discrete_B, W):
    """
    q = find_strategy_discrete(distribution1, distribution2, W)

    Find bayesian strategy for 2 discrete distributions.

    :param discrete_A['Prob']:      pXk(x|A) given as a <1 × n> np array
    :param discrete_A['Prior']:     prior probability pK(A)
    :param discrete_B['Prob']:      pXk(x|B) given as a <1 × n> np array
    :param discrete_B['Prior']:     prior probability pK(B)
    :param W:                       cost function matrix
                                    dims: <states x decisions>
                                    (nr. of states and decisions is fixed to 2)
    :return:                        q - optimal strategy <1 x n>, values 0 or 1
    """

    raise NotImplementedError("You have to implement this function.")
    q = None

    return q


def classify_discrete(imgs, q):
    """
    function label = classify_discrete(imgs, q)

    Classify images using discrete measurement and strategy q.

    :param imgs:    test set images, <h x w x n>
    :param q:       strategy <1 × 21> np array of 0 or 1
    :return:        image labels, <1 x n>
    """
    raise NotImplementedError("You have to implement this function.")
    label = None
    return label


def classification_error_discrete(images, labels, q):
    """
    error = classification_error_discrete(images, labels, q)

    Compute classification error for a discrete strategy q.

    :param images:      <h x w x n> np array of images
    :param labels:      <1 x n> np array of 0 or 1
    :param q:           <1 × m> np array of 0 or 1
    :return:            error - classification error as a fraction of false samples
                        scalar in range <0, 1>
    """
    raise NotImplementedError("You have to implement this function.")
    error = None
    return error


def find_strategy_2normal(distribution_A, distribution_B):
    """
    q = find_strategy_2normal(distribution_A, distribution_B)

    Find optimal bayesian strategy for 2 normal distributions and zero-one loss function.

    :param distribution_A:  parameters of the normal dist.
                            distribution_A['Mean'], distribution_A['Sigma'], distribution_A['Prior']
    :param distribution_B:  the same as distribution_A
    :return q:              strategy
%               q['t1'] q['t2'] - two descision thresholds
%               q['decision'] - 3 decisions for intervals (-inf, t1>, (t1, t2>, (t2, inf)
%                            shape <1 x 3>
    """
    raise NotImplementedError("You have to implement this function.")
    q = None
    return q


def bayes_risk_2normal(distribution_A, distribution_B, q):
    """
    R = bayes_risk_2normal(distribution_A, distribution_B, q)

    Compute bayesian risk of a strategy q for 2 normal distributions and zero-one loss function.

    :param distribution_A:  parameters of the normal dist.
                            distribution_A['Mean'], distribution_A['Sigma'], distribution_A['Prior']
    :param distribution_B:  the same as distribution_A
    :param q:               strategy
%                           q['t1'] q['t2'] - two descision thresholds
%                           q['decision'] - 3 decisions for intervals (-inf, t1>, (t1, t2>, (t2, inf)
%                           shape <1 x 3>
    :return:    R - bayesian risk, scalar
    """
    raise NotImplementedError("You have to implement this function.")
    R = None
    return R


def classify_2normal(imgs, q):
    """
    label = classify_2normal(imgs, q)

    Classify images using continuous measurement and strategy q.

    :param imgs:    test set images, <h x w x n>
    :param q:       strategy
%                   q['t1'] q['t2'] - two descision thresholds
%                   q['decision'] - 3 decisions for intervals (-inf, t1>, (t1, t2>, (t2, inf)
%                   shape <1 x 3>
    :return:        label - image labels, <1 x n>
    """
    raise NotImplementedError("You have to implement this function.")
    label = None
    return label


def classification_error_2normal(images, labels, q):
    """
    error = classification_error_2normal(images, labels, q)

    Compute classification error of a strategy q in a test set.

    :param images:  test set images, <h x w x n>
    :param labels:  test set labels <1 x n>
    :param q:       strategy
%                   q['t1'] q['t2'] - two descision thresholds
%                   q['decision'] - 3 decisions for intervals (-inf, t1>, (t1, t2>, (t2, inf)
%                   shape <1 x 3>
    :return:        classification error in range <0, 1>
    """
    raise NotImplementedError("You have to implement this function.")
    error = None
    return error



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

    :param imgs:    set of images, <h x w x n>
    :return:        measurements, <1 x n>
    """

    width = imgs.shape[1]
    sum_rows = np.sum(imgs, dtype=np.float64, axis=0)

    x = np.sum(sum_rows[0:int(width / 2),:],axis=0) - np.sum(sum_rows[int(width / 2):,:], axis=0)
    x = np.atleast_2d(x)
    # x = reshape(x,1,size(imgs,3));
    return x


def visualize_discrete(discrete_A, discrete_B, q):
    """
    visualize_discrete(discrete_A, discrete_B, q)

    Visualize a strategy for 2 discrete distributions.

    :param discrete_A['Prob']:      pXk(x|A) given as a <1 × n> np array
    :param discrete_A['Prior']:     prior probability pK(A)
    :param discrete_B['Prob']:      pXk(x|B) given as a <1 × n> np array
    :param discrete_B['Prior']:     prior probability pK(B)
    :param q:                       strategy - <1 × n> np array, values 1 or 2
    :return:
    """

    posterior_A = discrete_A['Prob'] * discrete_A['Prior']
    posterior_B = discrete_B['Prob'] * discrete_B['Prior']

    max_prob = np.max([posterior_A, posterior_B])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title("Posterior probabilities and strategy q")
    plt.xlabel("feature")
    plt.ylabel("posterior probabilities")

    bins = np.array(range(posterior_A.size + 1)) - int(posterior_A.size / 2)

    width = 0.75
    bar_plot_A = plt.bar(bins[:-1], posterior_A[0], width=width, color='b', alpha=0.75)
    bar_plot_B = plt.bar(bins[:-1], posterior_B[0], width=width, color='r', alpha=0.75)

    plt.legend((bar_plot_A, bar_plot_B), (r'$p_{XK}(x,A)$', r'$p_{XK}(x,B)$'))

    sub_level = - max_prob / 8
    height = np.abs(sub_level)
    for idx in range(len(bins[:-1])):
        b = bins[idx]
        col = 'r' if q[0, idx] == 1 else 'b'
        patch = patches.Rectangle([b - 0.5, sub_level], 1, height, angle=0.0, color=col, alpha=0.75)
        ax.add_patch(patch)

    plt.ylim(bottom=sub_level)
    plt.text(bins[0], -max_prob / 16, 'strategy q')


def compute_measurement_lr_discrete(imgs):
    """
    x = compute_measurement_lr_discrete(imgs)

    Calculates difference between left and right half of image(s).

    :param imgs:    set of images, <h x w x n> or <h x w x 3 x n> (color images)
    :return:        np array of values in range <-10, 10>,
                    shape <1 x n>
    """
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

    return x
