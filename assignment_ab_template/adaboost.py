import numpy as np
import matplotlib.pyplot as plt

def adaboost(X, y, num_steps):
    """
    Trains an AdaBoost classifier

    :param X:           <d x n> np.array with training data,
                            d - number of weak classifiers
                            n - number of data
    :param y:           <n x > np.array with training labels (-1 or 1)
    :param num_steps:   maximum number of iterations

    :return:            strong_classifier - dict with fields:
                            strong_classifier['wc'] - list of weak classifiers (see docstring of find_best_weak)
                            strong_classifier['alpha'] - list of weak classifier coefficients

    """
    raise NotImplementedError("You have to implement this function.")

    # initialisation
    N = y.size

    # prepare empty arrays for results
    strong_classifier = {'wc': [],
                         'alpha': []}

    for t in range(num_steps):
        pass

    return strong_classifier, wc_errors, upper_bound


def adaboost_classify(strong_classifier, X):
    """ Classifies data X with a strong classifier

    :param strong_classifier:   classifier returned by adaboost
    :param X:                   <d x n> np.array with training data,
                                    d - number of weak classifiers
                                    n - number of data

    :return:                    <n x > np.array of classification labels (values -1, 1)
    """
    raise NotImplementedError("You have to implement this function.")

    return classif


def compute_error(strong_classifier, X, y):
    """
    Computes the error on data X for all lengths of the given strong classifier

    :param strong_classifier:   classifier returned by adaboost (with T weak classifiers)
    :param X:                   <d x n> np.array with training data,
                                    d - number of weak classifiers
                                    n - number of data
    :param y:                   <n x > np.array with training labels (-1 or 1)

    :return:                    errors - <T x > np.array with errors of the strong classifier for all lengths from 1 to T
    """
    raise NotImplementedError("You have to implement this function.")

    return errors

################################################################################
#####                                                                      #####
#####             Below this line are already prepared methods             #####
#####                                                                      #####
################################################################################

def find_best_weak(X, y, D):
    """Finds best weak classifier

    Searches over all weak classifiers and their parametrisations
    (threshold and parity) for the weak classifier with lowest
    weighted classification error.

    The weak classifier realises following classification function:
        sign(parity * (x - theta))


    :param X:   <d x n> np.array with training data,
                    d - number of weak classifiers
                    n - number of data
    :param y:   <n x > np.array with training labels (-1 or 1)
    :param D:   <n x > np.array with training data weights

    :return:    wc - dict with fields:
                    wc['idx'] - index of the selected weak classifier
                    wc['theta'] - the classification threshold
                    wc['parity'] - the classification parity
                wc_error - the weighted error of the selected weak classifier
    """
    assert len(X.shape) == 2

    assert len(y.shape) == 1
    assert y.size == X.shape[1]

    assert len(D.shape) == 1
    assert D.size == X.shape[1]

    N_wc, N = X.shape

    best_err = np.inf

    wc = {}

    for i in range(N_wc):
        weak_X = X[i, :] # weak classifier evaluated on all data

        thresholds = np.unique(weak_X)
        assert len(thresholds.shape) == 1

        if thresholds.size > 1:
            thresholds = (thresholds[:-1] + thresholds[1:]) / 2.
        else:
            thresholds = np.array([+1, -1] + thresholds[0])
        assert len(thresholds.shape) == 1

        K = thresholds.size

        classif = np.sign(np.reshape(weak_X, (N, 1)) - np.reshape(thresholds, (1, K)))
        assert len(classif.shape) == 2
        assert classif.shape[0] == N
        assert classif.shape[1] == K

        column_D = np.reshape(D, (N, 1))
        column_y = np.reshape(y, (N, 1))

        err_pos = np.sum(column_D * (classif != column_y), axis=0)
        err_neg = np.sum(column_D * (-classif != column_y), axis=0)

        assert len(err_pos.shape) == 1
        assert err_pos.shape[0] == K
        assert len(err_neg.shape) == 1
        assert err_neg.shape[0] == K

        min_pos_idx = np.argmin(err_pos)
        min_pos_err = err_pos[min_pos_idx]

        min_neg_idx = np.argmin(err_neg)
        min_neg_err = err_neg[min_neg_idx]

        if min_pos_err < min_neg_err:
            err = min_pos_err
            parity = 1
            theta = thresholds[min_pos_idx]
        else:
            err = min_neg_err
            parity = -1
            theta = thresholds[min_neg_idx]

        if err < best_err:
            wc['idx'] = i
            wc['theta'] = theta
            wc['parity'] = parity

            best_err = err

    return wc, best_err



def montage(images, colormap='gray'):
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

def show_classification(test_images, labels):
    """
    show_classification(test_images, labels)

    create montages of images according to estimated labels

    :param test_images:     shape h x w x n
    :param labels:          shape n
    """
    imgs = test_images[..., labels == 1]
    subfig = plt.subplot(1, 2, 1)
    montage(imgs)
    plt.title('selected')

    imgs = test_images[..., labels == -1]
    subfig = plt.subplot(1, 2, 2)
    montage(imgs)
    plt.title('others')

def show_classifiers(class_images, classifier):
    """
    :param class_images:  <h x w x N> np.array of images of a selected number
    :param classifier:    adaboost classifier
    """
    assert len(class_images.shape) == 3

    mean_image = np.mean(class_images, axis=2)
    mean_image = np.dstack((mean_image, mean_image, mean_image))

    vis = np.reshape(mean_image, (-1, 3))
    max_alpha = np.amax(classifier['alpha'])

    for i, wc in enumerate(classifier['wc']):
        c = classifier['alpha'][i] / float(max_alpha)

        if wc['parity'] == 1:
            color = (c, 0, 0)
        else:
            color = (0, c, 0)

        vis[wc['idx'], :] = color

    vis = np.reshape(vis, mean_image.shape)

    plt.imshow(vis)
    plt.axis('off')
