#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import random
import itertools

def k_means(x, k, max_iter, show=False, init_means=None):
    """
    c, means, sq_dists = k_means(x, k, max_iter, show, init_means)

    Implementation of the k-means clustering algorithm.

    :param x:           Feature vectors, np.array of size <dim x number_of_vectors> (float64/double),
                        where dim is arbitrary feature vector dimension
    :param k:           Required number of clusters (scalar, int32)
    :param max_iter:    Stopping criterion: max. number of iterations (scalar, int32);
                        Set it to infinity (np.inf) if you wish to deactivate this criterion
    :param show:        Boolean switch to turn on/off visualization of partial
                        results, optional.
    :param init_means:  Initial cluster prototypes, np array <dim x k> (float64/double), optional
    :return:        c - Cluster index for each feature vector, np array of size
                        <1 x number_of_vectors>, containing only values from 1 to k,
                        i.e. c[i] is the index of a cluster which the vector x[:,i]
                        belongs to.
                means - Cluster centers, np array of size <dim x k> (float64/double),
                        i.e. means[:,i] is the center of the i-th cluster.
             sq_dists - Squared distances to the nearest mean for each feature vector,
                        np array of size <1 x number_of_vectors> (float64/double)

    Note 1: The iterative procedure terminates if either maximum number of iterations is reached
          or there is no change in assignment of data to the clusters.

    Note 2: DO NOT MODIFY INITIALIZATIONS

    """

    # Number of vectors
    N = x.shape[1]
    c = np.zeros([1, N], np.int32)

    # Means initialization
    if init_means is None:
        ind = random.sample(range(N), k)
        means = x[:, ind]
    else:
        means = init_means

    i_iter = 0
    while i_iter < max_iter:

        # YOUR CODE HERE
        raise NotImplementedError("You have to implement this function.")

        # Ploting partial results
        if show:
            print('Iteration: {:d}'.format(i_iter))
            show_clusters(x, c, means)

        if show:
            print('Done.')

    return c, means, sq_dists


def k_means_multiple_trials(x, k, n_trials, max_iter, show=False):
    """
    c, means, sq_dists = k_means_multiple_trials(x, k, n_trials, max_iter, show)

    Performs several trials of the k-means clustering algorithm in order to
    avoid local minima. Result of the trial with the lowest "within-cluster
    sum of squares" is selected as the best one and returned.

    :param x:           Feature vectors, np.array of size <dim x number_of_vectors> (float64/double),
                        where dim is arbitrary feature vector dimension
    :param k:           Required number of clusters (scalar, int32)
    :param n_trials:    Number of trials.
    :param max_iter:    Stopping criterion: max. number of iterations (scalar, int32);
                        Set it to infinity (np.inf) if you wish to deactivate this criterion
    :param show:        Boolean switch to turn on/off visualization of partial
                        results, optional.
    :return: (= information about the best clustering from all the trials):
                    c - Cluster index for each feature vector, np array of size
                        <1 x number_of_vectors>, containing only values from 1 to k,
                        i.e. c[i] is the index of a cluster which the vector x[:,i]
                        belongs to.
                means - Cluster centers, np array of size <dim x k> (float64/double),
                        i.e. means[:,i] is the center of the i-th cluster.
             sq_dists - Squared distances to the nearest mean for each feature vector,
                        np array of size <1 x number_of_vectors> (float64/double)
    """


    # YOUR CODE HERE

    # Multiple trial of the k-means clustering algorithm
    for i_trial in range(n_trials):

        # YOUR CODE HERE

        # Plotting partial results
        if show:
            # YOUR CODE HERE (use function show_clusters)
            raise NotImplementedError("You have to implement this function.")

    raise NotImplementedError("You have to implement this function.")
    return c, means, sq_dists


def random_sample(weights):
    """
    idx = random_sample(weights)

    picks randomly a sample based on the sample weights.
    Suppose weights / sum(weights) is a discrete probability distribution.

    :param weights: array of sample weights <1 x n> (float64/double)
    :return:        idx - index of chosen sample (scalar, int32)

    Note: use np.random.uniform() for random number generation in open interval (0, 1)
    """

    # use np.random.uniform() for random number generation in open interval (0, 1)

    raise NotImplementedError("You have to implement this function.")
    return idx


def k_meanspp(x, k):
    """
    centers = k_meanspp(x, k)

    perform k-means++ initialization for k-means clustering.

    :param x:   Feature vectors, np.array of size <dim x number_of_vectors> (float64/double),
                where dim is arbitrary feature vector dimension
    :param k:   Required number of clusters (scalar, int32)
    :return:    centers - k proposed centers for k-means initialization, np.array <dim x k> (float64/double)
    """

    N = x.shape[1]

    raise NotImplementedError("You have to implement this function.")
    return centers


def quantize_colors(im, k):
    """
    im_q = quantize_colors(im, k)

    Image color quantization using the k-means clustering. The pixel colors
    are at first clustered into k clusters. Color of each pixel is then set
    to the mean color of the cluster to which it belongs to.

    :param im:  Image whose pixel colors are to be quantized, np.array <h x w x 3> (uint8)
    :param k:   Required number of quantized colors (scalar, int32)
    :return:    Image with quantized colors, np.array <h x w x 3> (uint8)
    """

    # RGB -> LAB conversion is skipped due to server side missing module
    # Finish this task using rgb image

    # YOUR CODE HERE

    raise NotImplementedError("You have to implement this function.")

    # LAB -> RGB conversion is skipped due to server side missing module

    im_q = im_q.astype(dtype=np.uint8)
    return im_q



################################################################################
#####                                                                      #####
#####             Below this line are already prepared methods             #####
#####                                                                      #####
################################################################################


def compute_measurements(images):
    """
    X = compute_measurements(data)

    computes 2D features from image measurements

    :param data:          dict with keys 'images' and 'labels'
        - data['images'] - <H x W x N_images> np.uint8 array of images

    :return:              X - <2 x N_images> np.array of features
    """

    images = images.astype(np.float64)
    H, W, N = images.shape

    left = images[:, :(W//2), :]
    right = images[:, (W//2):, :]
    up = images[:(H//2), ...]
    down = images[(H//2):, ...]

    L = np.sum(left, axis=(0, 1))
    R = np.sum(right, axis=(0, 1))
    U = np.sum(up, axis=(0, 1))
    D = np.sum(down, axis=(0, 1))

    a = L - R
    b = U - D

    X = np.vstack((a, b))
    return X


def show_clusters(x, c, means):
    """
    show_clusters(x, c, means)

    Create plot of feature vectors with same colour for members of same cluster.

    :param x:       Feature vectors, np.array of size <dim x number_of_vectors> (float64/double),
                    where dim is arbitrary feature vector dimension (show only first 2 dimensions)
    :param c:       Cluster index for each feature vector, np array of size
                    <1 x number_of_vectors>, containing only values from 1 to k,
                    i.e. c[i] is the index of a cluster which the vector x[:,i]
                    belongs to.
    :param means:   Cluster centers, np array of size <dim x k> (float64/double),
                    i.e. means[:,i] is the center of the i-th cluster.
    """

    c = c.flatten()
    clusters = np.unique(c)
    markers = itertools.cycle(['*','o','+','x','v','^','<','>'])

    plt.figure()
    for i in clusters:
        cluster_x = x[:, c == i]
        # print(cluster_x)
        plt.plot(cluster_x[0], cluster_x[1], markers.next())
    plt.axis('equal')


    len = means.shape[1]
    for i in range(len):
        plt.plot(means[0,i], means[1,i], 'm+', ms=10, mew=2)


def show_clustered_images(images, labels):
    """
    show_clustered_images(test_images, labels[, letters])

    Shows results of clustering. Create montages of images according to estimated labels

    :param images:          input images, np.array <h x w x n>
    :param labels:          labels of input images, np.array <1 x n>
    """
    assert (len(images.shape) == 3)

    labels = labels.flatten()
    l = np.unique(labels)
    n = len(l)

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

    plt.figure(figsize=(10,10))
    ww = np.ceil(float(n) / np.sqrt(n))
    hh = np.ceil(float(n) / ww)

    for i in range(n):
        imgs = images[:, :, labels == unique_labels[i]]
        subfig = plt.subplot(hh,ww,i+1)
        montage(imgs)


def show_mean_images(images, labels, letters=None):
    """
    show_mean_images(images, c)

    Compute mean image for a cluster and show it.

    :param labels:       image labels, np.array of size <number of images>; <label> is index to <Alphabet>
    :param images:       images of letters, np.array of size <height x width x number of images>
    :return:             mean of all images of the <letter_char>, uint8 type
    """
    assert (len(images.shape) == 3)

    labels = labels.flatten()
    l = np.unique(labels)
    n = len(l)

    unique_labels = np.unique(labels).flatten()

    plt.figure()
    ww = np.ceil(float(n) / np.sqrt(n))
    hh = np.ceil(float(n) / ww)

    for i in range(n):
        imgs = images[:, :, labels == unique_labels[i]]
        img_average = np.squeeze(np.average(imgs.astype(np.float64), axis=2))
        subfig = plt.subplot(hh,ww,i+1)
        plt.imshow(img_average, cmap='gray')
        if letters is not None:
            plt.title(letters[i])


def gen_kmeanspp_data(mu=None, sigma=None, n=None):
    """
    samples = gen_kmeanspp_data()

    generates data with 4 normally distributed clusters

    :param mu:      mean of normal distribution (np.array)
    :param sigma:   std of normal distribution (scalar)
    :param n:       number of output points for each distribution
    :return:        samples - [2 x n*4] 2 dimensional samples with n samples per cluster
    """

    sigma = 1. if sigma is None else sigma
    mu = np.array([[-5, 0], [5, 0], [0, -5], [0, 5]]) if mu is None else mu
    n = 80 if n is None else n

    samples = np.random.normal(np.tile(mu, (n, 1)).T, sigma)
    return samples