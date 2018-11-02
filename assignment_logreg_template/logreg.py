#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def logistic_loss(X, y, w):
    """
    E = logistic_loss(X, y, w)

    Evaluates the logistic loss function.

    :param X:  d-dimensional observations, nd array <d x number_of_observations>
    :param y:  labels of the observations, nd array <1 x number_of_observations>
    :param w:  weights, nd array <d x 1>
    :return:   calculated loss (scalar)
    """
    raise NotImplementedError("You have to implement this function.")
    return E


def logistic_loss_gradient(X, y, w):
    """
    g = logistic_loss_gradient(X, y, w)

    Calculates gradient of the logistic loss function.

    :param X:  d-dimensional observations, nd array <d x number_of_observations>
    :param y:  labels of the observations, nd array <1 x number_of_observations>
    :param w:  weights, nd array <d x 1>
    :return:   g - resulting gradient vector, nd array <d x 1>
    """
    raise NotImplementedError("You have to implement this function.")
    return g


def logistic_loss_gradient_descent(X, y, w_init, epsilon):
    """
    w, wt, Et = logistic_loss_gradient_descent(X, y, w_init, epsilon)

    Performs gradient descent optimization of the logistic loss function.

    :param X:       d-dimensional observations, nd array <d x number_of_observations>
    :param y:       labels of the observations, nd array <1 x number_of_observations>
    :param w_init:  initial weights, nd array <d x 1>
    :param epsilon: parameter of termination condition: np.norm(w_new - w_prev) <= epsilon
    :return:        w - resulting weights, nd array <d x 1>
                    wt - progress of weights, nd array <d x number_of_accepted_candidates>
                    Et - progress of logistic loss, nd array <1 x number_of_accepted_candidates>
    """
    raise NotImplementedError("You have to implement this function.")
    return w, wt, Et


def classify_images(X, w):
    """
    y = classify_images(X, w)

    Classification by logistic regression.

    :param X: d-dimensional observations, nd array <d x number_of_observations>
    :param w: weights, nd array <d x 1>
    :return:  y - labels of the observations, nd array <1 x number_of_observations>
    """
    raise NotImplementedError("You have to implement this function.")
    return y


def get_threshold(w):
    """
    threshold = get_threshold(w)

    Returns the optimal decision threshold given the sigmoid parameters w

    :param w: sigmoid parameters np array <2 x 1>
    :return: calculated threshold (scalar)
    """
    raise NotImplementedError("You have to implement this function.")
    return threshold


################################################################################
#####                                                                      #####
#####             Below this line are already prepared methods             #####
#####                                                                      #####
################################################################################

def plot_gradient_descent(X, y, loss_function, w, wt, Et):
    """
    plot_gradient_descent(X, y, loss_function, w, wt, Et)

    Plots the progress of the gradient descent.

    :param X:               d-dimensional observations, nd array <d x number_of_observations>
    :param y:               labels of the observations, nd array <1 x number_of_observations>
    :param loss_function:   pointer to a logistic loss function
    :param w:               weights, nd array <d x 1>
    :param wt:              progress of weights, nd array <d x number_of_accepted_candidates>
    :param Et:              progress of logistic loss, nd array <1 x number_of_accepted_candidates>
    :return:
    """

    assert (len(X.shape) == 2)
    assert (len(y.shape) == 2) and (y.shape[0] == 1)
    assert (len(wt.shape) == 2)
    assert (len(w.shape) == 2) and (w.shape[1] == 1)
    assert (len(Et.shape) == 2) and (Et.shape[0] == 1)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca(projection='3d')

    # Plot loss function

    # Display range
    minW = -10
    maxW = 10

    points = 20
    if X.shape[0] == 1:
        pass
        # w = np.linspace(minW, maxW, points)
        # L = arrayfun(lambda weights: loss_function(X, y, weights), w)
        # plt.plot(w.T, L.T)
    elif X.shape[0] == 2:
        W1, W2 = np.meshgrid(np.linspace(minW, maxW, points), np.linspace(minW, maxW, points))
        L = np.zeros_like(W1)
        for i in range(W1.shape[0]):
            for j in range(W1.shape[1]):
                L[i, j] = loss_function(X, y, np.array([[W1[i, j], W2[i, j]]]).T)
        surf = ax.plot_surface(W1, W2, L, cmap='plasma')
        ax.view_init(90, 90)

        plt.title('Gradient descent')

        plt.colorbar(surf, ax=ax)
    else:
        raise NotImplementedError('Only 1-d and 2-d loss functions can be visualized using this method.')

    # Plot the gradient descent

    offset = 0.05

    # Highlight the found minimum
    ax.plot3D(w[0, :], w[1, :], logistic_loss(X, y, w) + offset, 'gs', markersize=15)

    ax.plot3D(wt[0, :], wt[1, :], Et.flatten() + offset, 'w.', markersize=15)
    ax.plot3D(wt[0, :], wt[1, :], Et.flatten() + offset, 'w-', markersize=15)

    plt.axis('equal')
    plt.xlim([np.min(wt[0, :]), np.max(wt[0, :])])
    plt.ylim([np.min(wt[1, :]), np.max(wt[1, :])])

    plt.xlabel('w_0')
    plt.ylabel('w_1')


def plot_aposteriori(X, y, w):
    """
    plot_aposteriori(X, y, w)

    :param X:               d-dimensional observations, nd array <d x number_of_observations>
    :param y:               labels of the observations, nd array <1 x number_of_observations>
    :param w:               weights, nd array <d x 1>
    """
    assert (len(X.shape) == 2)
    assert (len(y.shape) == 2) and (y.shape[0] == 1)
    assert (len(w.shape) == 2) and (w.shape[1] == 1)

    xA = X[y == 1]
    xC = X[y == -1]

    plot_range = np.linspace(np.min(X) - 0.5, np.max(X) + 0.5, 100)
    pAx = 1 / (1 + np.exp(-plot_range * w[1, 0] - w[0, 0]))
    pCx = 1 / (1 + np.exp(plot_range * w[1, 0] + w[0, 0]))

    thr = get_threshold(w)

    plt.figure()
    plt.plot(plot_range, pAx, 'b-', LineWidth=2)
    plt.plot(plot_range, pCx, 'r-', LineWidth=2)
    plt.plot(xA, np.zeros_like(xA), 'b+')
    plt.plot(xC, np.ones_like(xC), 'r+')
    plt.plot([thr, thr], [0, 1], 'k-')
    plt.legend(['p(A|x)', 'p(C|x)'])


def compute_measurements(imgs, norm_parameters=None):
    """
    x = compute_measurement(imgs [, norm_parameters])

    Compute measurement on images, subtract sum of right half from sum of
    left half.

    :param imgs:            set of images, <h x w x n>
    :param norm_parameters: [[mean, std]], np array <1 x 2>
    :return:                x - measurements, <1 x n>
                            norm_parameters - [[mean, std]], np array <1 x 2>
    """
    assert (len(imgs.shape) == 3)
    assert (norm_parameters is None) or (norm_parameters.shape == (1,2))

    width = imgs.shape[1]
    sum_rows = np.sum(imgs, dtype=np.float64, axis=0)

    x = np.sum(sum_rows[0:int(width / 2),:],axis=0) - np.sum(sum_rows[int(width / 2):,:], axis=0)
    x = np.atleast_2d(x)

    if norm_parameters is None:
        # If normalization parameters are not provided, compute it from data
        norm_parameters = np.array([np.mean(x), np.std(x)], np.float64)
    else:
        norm_parameters = norm_parameters.flatten()

    x = (x - norm_parameters[0]) / norm_parameters[1]
    norm_parameters = np.atleast_2d(norm_parameters)

    assert (norm_parameters.shape == (1,2))
    return x, norm_parameters


def show_classification(test_images, labels, letters):
    """
    show_classification(test_images, labels, letters)

    create montages of images according to estimated labels

    :param test_images:     shape h x w x n
    :param labels:          shape 1 x n
    :param letters:         string with letters, e.g. 'CN'
    """
    assert (len(test_images.shape) == 3)
    assert (len(labels.shape) == 2) and (labels.shape[0] == 1)

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

    unique_labels = np.unique(labels).flatten()
    for i in range(len(letters)):
        imgs = test_images[:,:,labels[0]==unique_labels[i]]
        subfig = plt.subplot(1,len(letters),i+1)
        montage(imgs)
        plt.title(letters[i])


def show_mnist_classification(imgs, labels, imsize=None):
    """
    function show_mnist_classification(imgs, labels)

    Shows results of MNIST digits classification.

    :param imgs:    set of images, 2D np array <(w x h) x n>
    :param labels:  estimated labels for the images, np array <1 x n>
    :param imsize:  estimated labels for the images, np array <1 x n>
    :return:
    """
    assert (len(imgs.shape) == 2)
    assert (len(labels.shape) == 2) and (labels.shape[0] == 1)
    assert (imsize is None) or ((len(imsize.shape) == 2) and (imsize.shape[0] == 1))

    if imsize is None:
        imsize = np.array([[28,28]])

    imsize = imsize.flatten()
    nImages = imgs.shape[1]
    images = np.zeros([imsize[0], imsize[1], nImages])

    for i in range(nImages):
        images[:, :, i] = np.reshape(imgs[:, i], [imsize[0], imsize[1]])

    plt.figure(figsize=(20,10))
    show_classification(images, labels, '01')