#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


def matrix_manip(A, B):
    """
    output = matrix_manip(A,B)

    Perform example matrix manipulations.

    :param A: matrix, arbitrary shape
    :param B: matrix, <2 x n>
    :return:
       output.A_transpose
       output.A_3rd_col
       output.A_slice
       output.A_gr_inc
       output.C
       output.A_weighted_col_sum
       output.D
       output.D_select

    """

    raise NotImplementedError("You have to implement this function.")
    output = None
    return output


def compute_letter_mean(letter_char, alphabet, images, labels):
    """
    img = compute_letter_mean(letter_char, alphabet, images, labels)

    Compute mean image for a letter.

    :param letter_char:  character, e.g. 'm'
    :param alphabet:     np.array of all characters present in images <n_letters x >
    :param images:       images of letters, np.array of size <height x width x number of images>
    :param labels:       image labels, np.array of size <number of images>; <label> is index to <Alphabet>
    :return:             mean of all images of the <letter_char>, uint8 type
    """
    raise NotImplementedError("You have to implement this function.")
    letter_mean = None
    return letter_mean


def compute_lr_histogram(letter_char, alphabet, images, labels, num_bins, return_bin_edges=False):
    """
    lr_histogram = compute_lr_histogram(letter_char, alphabet, images, labels, num_bins)

    Calculates feature histogram.

    :param letter_char:                is a character representing the letter whose feature histogram
                                       we want to compute, e.g. 'C'
    :param alphabet:                   np.array of all characters present in images <n_letters x >
    :param images:                     images in 3d matrix of shape <h x w x n>
    :param labels:                     labels of images, indices to Alphabet list, <n x >
    :param num_bins:                   number of histogram bins
    :param return_bin_edges:
    :return:                           counts of values in the corresponding bins, np.array <num_bins, >,
                                       bin edges, if return_bin_edges is True, np.array <num_bins+1, >
    """

    raise NotImplementedError("You have to implement this function.")
    if return_bin_edges:
        return lr_histogram, bin_edges
    else:
        return lr_histogram


def histogram_plot(hist_data, color, alpha):
    """
    Plot histogram from outputs of compute_lr_histogram

    :param hist_data: tuple of (histogram values, histogram bin edges)
    :param color:     color of the histogram (passed to matplotlib)
    :param alpha:     transparency alpha of the histogram (passed to matplotlib)
    """

    raise NotImplementedError("You have to implement this function.")

################################################################################
#####                                                                      #####
#####             Below this line are already prepared methods             #####
#####                                                                      #####
################################################################################

def montage(images, colormap='gray'):
    """
    Show images in grid.

    :param images:  np.array <h x w x n>
    """
    h, w, count = np.shape(images)
    h_sq = np.int(np.ceil(np.sqrt(count)))
    w_sq = h_sq
    im_matrix = np.zeros((h_sq * h, w_sq * w))

    image_id = 0
    for k in range(w_sq):
        for j in range(h_sq):
            if image_id >= count:
                break
            slice_w = j * h
            slice_h = k * w
            im_matrix[slice_h:slice_h + w, slice_w:slice_w + h] = images[:, :, image_id]
            image_id += 1
    plt.imshow(im_matrix, cmap=colormap)
    plt.axis('off')
    return im_matrix
