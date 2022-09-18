#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


def matrix_manip(A, B):
    """
    output = matrix_manip(A,B)

    Perform example matrix manipulations.

    :param A: np.array (k, l), l >= 3
    :param B: np.array (2, n)
    :return:
       output['A_transpose'] (l, k), same dtype as A
       output['A_3rd_col'] (k, 1), same dtype as A
       output['A_slice'] (2, 3), same dtype as A
       output['A_gr_inc'] (k, l+1), same dtype as A
       output['C'] (k, k), same dtype as A
       output['A_weighted_col_sum'] python float
       output['D'] (2, n), same dtype as B
       output['D_select'] (2, n'), same dtype as B

    """

    raise NotImplementedError("You have to implement this function.")
    output = None
    return output


def compute_letter_mean(letter_char, alphabet, images, labels):
    """
    img = compute_letter_mean(letter_char, alphabet, images, labels)

    Compute mean image for a letter.

    :param letter_char:  character, e.g. 'm'
    :param alphabet:     np.array of all characters present in images (n_letters, )
    :param images:       images of letters, np.array of size (H, W, n_images)
    :param labels:       image labels, np.array of size (n_images, ) (index into alphabet array)
    :return:             mean of all images of the letter_char, (H, W) np.uint8 dtype (round, then convert)
    """
    raise NotImplementedError("You have to implement this function.")
    letter_mean = None
    return letter_mean


def compute_lr_features(letter_char, alphabet, images, labels):
    """
    lr_features = compute_lr_features(letter_char, alphabet, images, labels)

    Calculates LR features for all letters.

    :param letter_char:                is a character representing the letter whose feature histogram
                                       we want to compute, e.g. 'C'
    :param alphabet:                   np.array of all characters present in images (n_letters, )
    :param images:                     images of letters, np.array of shape (h, w, n_images)
    :param labels:                     image_labels, np.array of size (n_images, ) (indexes into alphabet array)
    :return:                           features for all occurrences of specific :param letter_char:, np.array of shape (n_letter_occurrences, )
    """

    raise NotImplementedError("You have to implement this function.")
    return lr_features


################################################################################
#####                                                                      #####
#####             Below this line are already prepared methods             #####
#####                                                                      #####
################################################################################


def plot_letter_feature_histogram(features_1, features_2, letters, n_bins=20):
    """
    Plot histograms from two sets of precomputed features
    :param features_1: features for all occurrences of specific letter 1, np.array of shape (n_letter_1, )
    :param features_2: features for all occurrences of specific letter 2, np.array of shape (n_letter_2, )
    :param letters:    are a characters representing letters which feature histogram we want to compute, e.g. 'CZ', string of the length = 2
    :param n_bins:     number of histogram bins, integer
    :return:
    """
    plt.figure()
    plt.title("letter feature histogram")
    plt.xlabel("LR feature")
    plt.ylabel("# Images")
    plt.hist(features_1, bins=n_bins, histtype='bar', edgecolor='black', linewidth=1.1, alpha=1., label=f'letter {letters[0]}')
    plt.hist(features_2, bins=n_bins, histtype='bar', edgecolor='black', linewidth=1.1, alpha=0.5, label=f'letter {letters[1]}')
    plt.legend()

def plot_letter_feature_histogram_interactive(alphabet, images, labels, n_bins=(2, 40)):
    """
    Interactive version of plot_letter_feature_histogram()

    You have to have installed ipywidgets to run interactive methods.

    :param alphabet: np.array of all characters present in images (n_letters, )
    :param images:   images of letters, np.array of shape (h, w, n_images)
    :param labels:   image_labels, np.array of size (n_images, ) (indexes into alphabet array)
    :param n_bins:   number of histogram bins, integer
    :return:
    """
    try:
        from ipywidgets import interact, interactive, fixed

        @interact(letterA=alphabet, letterB=alphabet, n_bins=n_bins)
        def plot_interactive_letter_feature_histogram(letterA='R', letterB='A', n_bins=20):
            features_1 = compute_lr_features(str(letterA), alphabet, images, labels)
            features_2 = compute_lr_features(str(letterB), alphabet, images, labels)
            plot_letter_feature_histogram(features_1, features_2, letterA + letterB, n_bins=n_bins)

    except ImportError:
        print('Optional feature.')

def plot_letter_mean_interactive(alphabet, images, labels):
    """
    Interactive version of compute_letter_mean()

    You have to have installed ipywidgets to run interactive methods.

    :param alphabet: np.array of all characters present in images (n_letters, )
    :param images:   images of letters, np.array of shape (h, w, n_images)
    :param labels:   image_labels, np.array of size (n_images, ) (indexes into alphabet array)
    :return:
    """
    try:
        from ipywidgets import interact, interactive, fixed

        @interact(letter=alphabet)
        def plot_interactive_letter_mean(letter='R'):
            initial_mean_interactive = compute_letter_mean(str(letter), alphabet, images, labels)
            plt.title("{} mean".format(letter))
            plt.imshow(initial_mean_interactive, cmap='gray');

    except ImportError:
        print('Optional feature.')


def montage(images, colormap='gray'):
    """
    Show images in grid.

    :param images:  np.array (h x w x n_images)
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

################################################################################
#####                                                                      #####
#####             Below this line you may insert debugging code            #####
#####                                                                      #####
################################################################################

def main():
    # HERE IT IS POSSIBLE TO ADD YOUR TESTING OR DEBUGGING CODE
    pass

if __name__ == "__main__":
    main()