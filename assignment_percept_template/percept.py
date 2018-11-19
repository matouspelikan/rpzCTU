#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage


def perceptron(X, y, maxIterations):
    """
    w, b = perceptron(X, y, maxIterations)

    Perceptron algorithm.
    Implements the perceptron algorithm
    (http://en.wikipedia.org/wiki/Perceptron)

    :param X:               d-dimensional observations, np array <d x number_of_observations>
    :param y:               labels of the observations (0 or 1), np array <1 x number_of_observations>
    :param maxIterations:   number of algorithm iterations (scalar)
    :return:                w - weights, np array <d x 1>
                            b - bias, (scalar)
    """
    raise NotImplementedError("You have to implement this function")
    return w, b


def lift_dimension(X):
    """
    Z = lift_dimension(X)

    Lifts the dimensionality of the feature space from 2 to 5 dimensions
    Z = (x_1 , x_2 , (x_1)^2 , x_1 * x_2 , (x_2)^2)

    :param X:   observations in the original space, np array <2 x number_of_observations>
    :return:    Z - observations in the lifted feature space, np array <5 x number_of_observations>
    """
    raise NotImplementedError("You have to implement this function")
    return Z


def classif_quadrat_perc(tst, model):
    """
    K = classif_quadrat_perc(tst, model)

    Classifies test samples using the quadratic discriminative function

    :param tst:     observations for classification in original space <2 x number_of_samples>
    :param model:   structure with the trained perceptron classifier
                    (parameters of the discriminative function)
                    model['w'] - weights vector, np array <d x 1>
                    model['b'] - bias term (1 double)
    :return:        Y - classification result (contains either 0 or 1), np array <1 x number_of_samples>
    """
    raise NotImplementedError("You have to implement this function")
    return Y


################################################################################
#####                                                                      #####
#####             Below this line are already prepared methods             #####
#####                                                                      #####
################################################################################

def pboundary(X, y, model):
    """
    pboundary(X, y, model)

    Plot boundaries for perceptron decision strategy

    :param X:       d-dimensional observations, np array <d x number_of_observations>
    :param y:       labels of the observations (0 or 1), np array <1 x number_of_observations>
    :param model:   structure with the trained perceptron classifier
                    (parameters of the discriminative function)
                    model['w'] - weights vector, np array <d x 1>
                    model['b'] - bias term (1 double)
    """

    assert (len(X.shape) == 2)
    assert (len(y.shape) == 2) and (y.shape[0] == 1)
    assert ('w' in model) and ('b' in model)
    assert (len(model['w'].shape) == 2) and (model['w'].shape[1] == 1)

    plt.figure()
    yf=y.flatten()
    plt.plot(X[0,yf==0],X[1,yf==0],'bx',ms=10)
    plt.plot(X[0,yf==1],X[1,yf==1],'r+',ms=10)

    minx, maxx = plt.xlim()
    miny, maxy = plt.ylim()

    epsilon = 0.1 * np.maximum(np.abs(maxx - minx), np.abs(maxy - miny))

    x_space = np.linspace(minx-epsilon, maxx+epsilon, 1000)
    y_space = np.linspace(miny-epsilon, maxy+epsilon, 1000)
    x_grid, y_grid = np.meshgrid(x_space, y_space)

    x_grid_fl = x_grid.reshape([1, -1])
    y_grid_fl = y_grid.reshape([1, -1])

    X_grid = np.concatenate([x_grid_fl, y_grid_fl], axis=0)
    Y_grid = classif_quadrat_perc(X_grid, model)
    Y_grid = Y_grid.reshape([1000,1000])

    blurred_Y_grid = ndimage.gaussian_filter(Y_grid, sigma=0)

    plt.contour(x_grid, y_grid, blurred_Y_grid, colors=['black'])
    plt.xlim(minx, maxx)
    plt.ylim(miny, maxy)
