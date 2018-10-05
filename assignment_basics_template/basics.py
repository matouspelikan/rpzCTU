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

    output = {}

    # 1. Find the transpose of the matrix A:
    output['A_transpose'] = None # TODO

    # 2. Select the third column of the matrix A:
    output['A_3rd_col'] = None # TODO

    # 3. Select last two rows and last three columns of the matrix A and return the matrix in output.A_slice.
    output['A_slice'] = None # TODO

    # 4.Find all positions in A greater then 3 and increment them by 1 and add a column of ones to the matrix.
    # Save the result to matrix A_gr_inc:
    output['A_gr_inc'] = None # TODO

    # 5. Create matrix C such that Ci,j=∑nk=1A_gr_inci,k⋅A_gr_incTk,j and store it in output.C.
    output['C'] = None # TODO

    # 6. Compute ∑nc=1c⋅∑mr=1A_gr_incr,c:
    output['A_weighted_col_sum'] = None # TODO

    # 7. Subtract a vector (4,6)T from all columns of matrix B. Save the result to matrix output.D.
    output['D'] = None # TODO

    # 8. Select all vectors in the matrix D, which have greater euclidean distance than the average euclidean distance.
    output['D_select'] = None # TODO
    return output


def compute_letter_mean(letter_char, alphabet, images, labels):
    """
    img = compute_letter_mean(letter_char, alphabet, images, labels)

    Compute mean image for a letter.

    :param letter_char:  character, e.g. 'm'
    :param alphabet:     list of all characters present in images, e.g. 'abcdefgh'
    :param images:       images of letters, matrix of size (height x width x number of images) (numpy.array)
    :param labels:       image labels, vector of size <number of images>; <label> is index to <Alphabet>
    :return:             mean of all images of the <letter_char>, HxW, rounded to uint8 type
    """

    letter_mean = None # TODO
    return letter_mean


def compute_lr_histogram(letter_char, alphabet, images, labels, num_bins, return_bin_edges=False):
    """
    lr_histogram = compute_lr_histogram(letter_char, alphabet, images, labels, num_bins)

    Calculates feature histogram.

    :param letter_char:                is a character representing the letter whose feature histogram
                                       we want to compute, e.g. 'C'
    :param alphabet:                   string of characters
    :param images:                     images in 3d matrix of shape <h x w x n>
    :param labels:                     labels of images, indices to Alphabet list, <1 x n>
    :param num_bins:                   number of histogram bins
    :param return_bin_edges:           toggle additional output
    :return:                           counts of values in the corresponding bins, vector <1 x num_bins>
    :return:                           (optional if return_bin_edges) vector of bin edges, vector <1 x num_bins + 1>
    """

    lr_histogram = None # TODO
    if return_bin_edges:
        return lr_histogram, bin_edges
    else:
        return lr_histogram
