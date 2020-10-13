import numpy as np
from scipy.stats import norm
import scipy.special as spec  # for gamma
# importing bayes doesn't work in BRUTE :(, please copy the functions into this file


# MLE
def ml_estim_normal(x):
    """
    Computes maximum likelihood estimate of mean and variance of a normal distribution.

    :param x:   measurements, numpy array (n, )
    :return:    mu - mean - python float
                var - variance - python float
    """
    raise NotImplementedError("You have to implement this function.")
    mu, var = None, None
    return mu, var


def ml_estim_categorical(counts):
    """
    Computes maximum likelihood estimate of categorical distribution parameters.

    :param counts: measured bin counts, numpy array (n, )
    :return:       pk - parameters of the categorical distribution, numpy array (n, )
    """
    raise NotImplementedError("You have to implement this function.")
    pk = None
    return pk

# MAP
def map_estim_normal(x, mu0, nu, alpha, beta):
    """
    Maximum a posteriori parameter estimation of normal distribution with normal inverse gamma prior.

    :param x:      measurements, numpy array (n, )
    :param mu0:    NIG parameter - python float
    :param nu:     NIG parameter - python float
    :param alpha:  NIG parameter - python float
    :param beta:   NIG parameter - python float

    :return:       mu - estimated mean - python float,
                   var - estimated variance - python float
    """
    raise NotImplementedError("You have to implement this function.")
    mu, var = None, None
    return mu, var


def map_estim_categorical(counts, alpha):
    """
    Maximum a posteriori parameter estimation of categorical distribution with Dirichlet prior.

    :param counts:  measured bin counts, numpy array (n, )
    :param alpha:   Dirichlet distribution parameters, numpy array (n, )

    :return:        pk - estimated categorical distribution parameters, numpy array (n, )
    """
    raise NotImplementedError("You have to implement this function.")
    pk = None
    return pk

# BAYES
def bayes_posterior_params_normal(x, prior_mu0, prior_nu, prior_alpha, prior_beta):
    """
    Compute a posteriori normal inverse gamma parameters from data and NIG prior.

    :param x:            measurements, numpy array (n, )
    :param prior_mu0:    NIG parameter - python float
    :param prior_nu:     NIG parameter - python float
    :param prior_alpha:  NIG parameter - python float
    :param prior_beta:   NIG parameter - python float

    :return:             mu0:    a posteriori NIG parameter - python float
    :return:             nu:     a posteriori NIG parameter - python float
    :return:             alpha:  a posteriori NIG parameter - python float
    :return:             beta:   a posteriori NIG parameter - python float
    """
    raise NotImplementedError("You have to implement this function.")
    mu0, nu, alpha, beta = None, None, None, None
    return mu0, nu, alpha, beta

def bayes_posterior_params_categorical(counts, alphas):
    """
    Compute a posteriori Dirichlet parameters from data and Dirichlet prior.

    :param counts:   measured bin counts, numpy array (n, )
    :param alphas:   prior Dirichlet distribution parameters, numpy array (n, )

    :return:         posterior_alphas - estimated Dirichlet distribution parameters, numpy array (n, )
    """
    raise NotImplementedError("You have to implement this function.")
    posterior_alphas = None
    return posterior_alphas

def bayes_estim_pdf_normal(x_test, x,
                           mu0, nu, alpha, beta):
    """
    Compute pdf of predictive distribution for Bayesian estimate for normal distribution with normal inverse gamma prior.

    :param x_test:  values where the pdf should be evaluated, numpy array (m, )
    :param x:       'training' measurements, numpy array (n, )
    :param mu0:     prior NIG parameter - python float
    :param nu:      prior NIG parameter - python float
    :param alpha:   prior NIG parameter - python float
    :param beta:    prior NIG parameter - python float

    :return:        pdf - Bayesian estimate pdf evaluated at x_test, numpy array (m, )
    """
    raise NotImplementedError("You have to implement this function.")
    pdf = None
    return pdf

def bayes_estim_categorical(counts, alphas):
    """
    Compute parameters of Bayesian estimate for categorical distribution with Dirichlet prior.

    :param counts:  measured bin counts, numpy array (n, )
    :param alphas:  prior Dirichlet distribution parameters, numpy array (n, )

    :return:        pk - estimated categorical distribution parameters, numpy array (n, )
    """
    raise NotImplementedError("You have to implement this function.")
    pk = None
    return pk

# Classification
def mle_Bayes_classif(test_imgs, train_data_A, train_data_C):
    """
    Classify images using Bayes classification using MLE of normal distributions and 0-1 loss.

    :param test_imgs:      images to be classified, numpy array (H, W, N)
    :param train_data_A:   training image features A, numpy array (nA, )
    :param train_data_C:   training image features C, numpy array (nC, )

    :return:               q - classification strategy (see find_strategy_2normal)
    :return:               labels - classification of test_imgs, numpy array (N, ) (see bayes.classify_2normal)
    :return:               DA - parameters of the normal distribution of A
                            DA['Mean'] - python float
                            DA['Sigma'] - python float
                            DA['Prior'] - python float
    :return:               DC - parameters of the normal distribution of C
                            DC['Mean'] - python float
                            DC['Sigma'] - python float
                            DC['Prior'] - python float
    """
    raise NotImplementedError("You have to implement this function.")
    q, labels, DA, DC = None, None, None, None
    return q, labels, DA, DC


def map_Bayes_classif(test_imgs, train_data_A, train_data_C,
                      mu0_A, nu_A, alpha_A, beta_A,
                      mu0_C, nu_C, alpha_C, beta_C):
    """
    Classify images using Bayes classification using MAP estimate of normal distributions with NIG priors and 0-1 loss.

    :param test_imgs:      images to be classified, numpy array (H, W, N)
    :param train_data_A:   training image features A, numpy array (nA, )
    :param train_data_C:   training image features C, numpy array (nC, )

    :param mu0_A:          prior NIG parameter for A - python float
    :param nu_A:           prior NIG parameter for A - python float
    :param alpha_A:        prior NIG parameter for A - python float
    :param beta_A:         prior NIG parameter for A - python float

    :param mu0_C:          prior NIG parameter for C - python float
    :param nu_C:           prior NIG parameter for C - python float
    :param alpha_C:        prior NIG parameter for C - python float
    :param beta_C:         prior NIG parameter for C - python float

    :return:               q - classification strategy (see find_strategy_2normal)
    :return:               labels - classification of test_imgs, numpy array (N, ) (see bayes.classify_2normal)
    :return:               DA - parameters of the normal distribution of A
                            DA['Mean'] - python float
                            DA['Sigma'] - python float
                            DA['Prior'] - python float
    :return:               DC - parameters of the normal distribution of C
                            DC['Mean'] - python float
                            DC['Sigma'] - python float
                            DC['Prior'] - python float
    """
    raise NotImplementedError("You have to implement this function.")
    q, labels, DA, DC = None, None, None, None
    return q, labels, DA, DC


def bayes_Bayes_classif(x_test, x_train_A, x_train_C,
                        mu0_A, nu_A, alpha_A, beta_A,
                        mu0_C, nu_C, alpha_C, beta_C):
    """
    Classify images using Bayes classification (0-1 loss) using predictive pdf estimated using Bayesian inferece with with NIG priors.

    :param x_test:         images features to be classified, numpy array (n, )
    :param x_train_A:      training image features A, numpy array (nA, )
    :param x_train_C:      training image features C, numpy array (nC, )

    :param mu0_A:          prior NIG parameter for A - python float
    :param nu_A:           prior NIG parameter for A - python float
    :param alpha_A:        prior NIG parameter for A - python float
    :param beta_A:         prior NIG parameter for A - python float

    :param mu0_C:          prior NIG parameter for C - python float
    :param nu_C:           prior NIG parameter for C - python float
    :param alpha_C:        prior NIG parameter for C - python float
    :param beta_C:         prior NIG parameter for C - python float

    :return:               labels - classification of x_test, numpy array (n, ) int32, values 0 or 1
    """
    raise NotImplementedError("You have to implement this function.")
    labels = None
    return labels


################################################################################
#####                                                                      #####
#####                Put functions from previous labs here.                #####
#####            (Sorry, we know imports would be much better)             #####
#####                                                                      #####
################################################################################


def find_strategy_2normal(distribution_A, distribution_B):
    """
    q = find_strategy_2normal(distribution_A, distribution_B)

    Find optimal bayesian strategy for 2 normal distributions and zero-one loss function.

    :param distribution_A:  parameters of the normal dist.
                            distribution_A['Mean'], distribution_A['Sigma'], distribution_A['Prior'] - python floats
    :param distribution_B:  the same as distribution_A

    :return q:              strategy dict
                               q['t1'], q['t2'] - decision thresholds - python floats
                               q['decision'] - (3, ) np.int32 np.array decisions for intervals (-inf, t1>, (t1, t2>, (t2, inf)
                               If there is only one threshold, q['t1'] should be equal to q['t2'] and the middle decision should be 0
                               If there is no threshold, q['t1'] and q['t2'] should be -/+ infinity and all the decision values should be the same (0 preferred)
    """
    raise NotImplementedError("You have to implement this function.")
    q = None

    return q


def classify_2normal(imgs, q):
    """
    label = classify_2normal(imgs, q)

    Classify images using continuous measurement and strategy q.

    :param imgs:    test set images, np.array (h, w, n)
    :param q:       strategy
                    q['t1'] q['t2'] - float decision thresholds
                    q['decision'] - (3, ) int32 np.array decisions for intervals (-inf, t1>, (t1, t2>, (t2, inf)
    :return:        label - image labels, (n, ) int32
    """
    raise NotImplementedError("You have to implement this function.")
    label = None
    return label


def classification_error_2normal(images, labels, q):
    """
    error = classification_error_2normal(images, labels, q)

    Compute classification error of a strategy q in a test set.

    :param images:  test set images, (h, w, n)
    :param labels:  test set labels (n, )
    :param q:       strategy
                    q['t1'] q['t2'] - float decision thresholds
                    q['decision'] - (3, ) np.int32 decisions for intervals (-inf, t1>, (t1, t2>, (t2, inf)
    :return:        python float classification error in range <0, 1>. Fraction of incorrect classifications.
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

    :param imgs:    set of images, (h, w, n) numpy array
    :return x:      measurements, (n, ) numpy array
    """
    assert len(imgs.shape) == 3

    width = imgs.shape[1]
    sum_rows = np.sum(imgs, dtype=np.float64, axis=0)

    x = np.sum(sum_rows[0:int(width / 2),:], axis=0) - np.sum(sum_rows[int(width / 2):,:], axis=0)

    assert x.shape == (imgs.shape[2], )
    return x


def mle_likelihood_normal(x, mu, var):
    """
    Compute the likelihood of the data x given the model is a normal distribution with given mean and sigma

    :param x:       measurements, numpy array (n, )
    :param mu:      the normal distribution mean
    :param var:     the normal distribution variance
    :return:        L - likelihood of the data x
    """
    assert len(x.shape) == 1

    if var <= 0:
        L = 0
    else:
        L = np.prod(norm.pdf(x, mu, np.sqrt(var)))
    return L


def norm_inv_gamma_pdf(mu, var, mu0, nu, alpha, beta):
    # Wikipedia sometimes uses a symbol 'lambda' instead 'nu'

    assert alpha > 0
    assert nu > 0
    if beta <= 0 or var <= 0:
        return 0

    sigma = np.sqrt(var)

    p = np.sqrt(nu) / (sigma * np.sqrt(2 * np.pi)) * np.power(beta, alpha) / spec.gamma(alpha) * np.power(1/var, alpha + 1) * np.exp(-(2 * beta + nu * (mu0 - mu) * (mu0 - mu)) / (2 * var))

    return p
