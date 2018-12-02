import numpy as np
import matplotlib.pyplot as plt

def get_kernel(Xi, Xj, options):
    """
    Returns kernel matrix K(Xi, Xj)

    :param Xi:       < d x m > np.array with features in columns
    :param Xj:       < d x n > np.array with features in columns
    :param options:  dict with options
                    options['kernel'] - one of 'rbf', 'linear', 'polynomial'
                    options['sigma'] - sigma for rbf kernel (no need to specify if kernel not rbf)
                    options['d'] - polynom degree for polynomial kernel (no need to specify if kernel not polynomial)

    :return:         < m x n > np.array with kernel function values
    """
    raise NotImplementedError("You have to implement this function.")

    return K

def my_kernel_svm(X, y, C, options):
    """
    Solves kernel soft-margin SVM dual task and outputs model

    :param X: < d x n >  np.array with features in columns
    :param y: < 1 x n >  np.array with labels (-1, 1) for X
    :param C:            scalar regularization constant C
    :param options:      dict with options for gsmo solver ('verb', 'tmax')
                           and get_kernel ('kernel', 'sigma'/'d')

    :return:           model - dict with entries:
                         model['sv'] - < d x n_sv > np.array of support vectors
                         model['y'] - < 1 x n_sv > np.array of support vector labels
                         model['alpha'] - < 1 x n_sv > np.array of support vector lagrange multipliers
                         model['options'] - kernel options (same as input)
                         model['b'] - scalar bias term
                         model['fun'] - function that should be used for classification
    """
    raise NotImplementedError("You have to implement this function.")

    return model

def classif_kernel_svm(X, model):
    """
    Performs classification on X by trained SVM classifier stored in model

    :param X:       < d x n > np.array of features in columns
    :param model:   dict with SVM model (my_kernel_svm output)

    :return:        classif - < 1 x n > np.array with classification labels (-1, 1)
    """
    raise NotImplementedError("You have to implement this function.")

    return classif

def compute_kernel_test_error(itrn, itst, X, y, C, options):
    """
    Computes mean risk computed over crossvalidation folds (train svm on train set and evaluate on test set)

    :param itrn:     list with training data indices folds (from crossval function)
    :param itst:     list with testing data indices folds (from crossval function)
    :param X:        <d x n> np.array - input feature vector
    :param y:        <1 x n> np.array - labels 1, -1
    :param C:        scalar regularization constant C
    :param options:  dict with options for my_kernel_svm and classif_kernel_svm

    :return:         mean test error
    """
    raise NotImplementedError("You have to implement this function.")

    return mean_test_error

################################################################################
#####                                                                      #####
#####             Below this line are already prepared methods             #####
#####                                                                      #####
################################################################################

def gsmo(H, f, a, b=0, lb=-np.inf, ub=np.inf, x0=None, nabla0=None, tolKKT=0.001, verb=0, tmax=np.inf):
    """
    GSMO Generalized SMO algorithm for classifier design.

    Translated to Python from:
    https://gitlab.fel.cvut.cz/smidm/statistical-pattern-recognition-toolbox

    Description:
     This function implements the generalized SMO algorithm which solves
      the following QP task:

      min Q_P(x) = 0.5*x'*H*x + f'*x
       x

      s.t.    a'*x = b
              lb(i) <= x(i) <= ub(i)   for all i=1:n

     Reference:
      S.S.Keerthi, E.G.Gilbert: Convergence of a generalized SMO algorithm for SVM
      classifier design. Machine Learning, Vol. 46, 2002, pp. 351-360.

    Input:
     H [n x n] Symmetric positive semidefinite matrix.
     f [n x 1] Vector.
     a [n x 1] Vector which must not contain zero entries.
     b [1 x 1] Scalar; default: 0
     lb [n x 1] Lower bound; default: -inf
     ub [n x 1] Upper bound; default: inf
     x0 [n x 1] Initial solution;
     nabla0 [n x 1] Nabla0 = H*x0 + f.
     tolKKT [1 x 1] Determines relaxed KKT conditions (default tolKKT=0.001);
                    it correspondes to $\tau$ in Keerthi's paper.
     verb [1 x 1] if > 0 then prints info every verb-th iterations (default 0)
     tmax [1 x 1] Maximal number of iterations (default inf).

    Output:
     tuple(
       x [n x 1] Solution vector.
       fval [1 x 1] Atteined value of the optimized QP criterion fval=Q_P(x);
       t [1x1] Number of iterations.
       finished [1x1] Has found an optimal solution
     )
     """
    N = H.shape[0]
    assert H.shape[1] == N
    assert f.shape[0] == N
    assert a.shape[0] == N

    # Setup
    if (np.isscalar(lb)):
        lb = np.array([lb] * len(f))

    if (np.isscalar(ub)):
        ub = np.array([ub] * len(f))

    if (x0 is None):
        # Find feasible x0
        x0 = np.zeros(len(f))
        xa = 0;
        i = 0
        while (not np.allclose(np.dot(x0, a), b)):
            if (i >= len(a)):
                raise ValueError("Constraints cannot be satisfied")
            x0[i] = np.clip((b - xa) / a[i], lb[i], ub[i])
            xa += x0[i] * a[i]
            i += 1
    else:
        x0 = np.clip(x0, lb, ub)

    if (nabla0 is None):
        nabla0 = np.dot(H, x0) + f

    # Initialization
    t = 0
    finished = False
    x = np.copy(x0)
    nabla = np.copy(nabla0)

    # SMO
    while (t < tmax):
        assert np.allclose(x, np.clip(x, lb, ub))  # x0 within bounds

        # Find the most violating pair of variables
        (minF, minI) = (np.inf, 0)
        (maxF, maxI) = (-np.inf, 0)
        a = np.squeeze(a)
        nabla = np.squeeze(nabla)
        x = np.squeeze(x)
        lb = np.squeeze(lb)
        ub = np.squeeze(ub)
        F = nabla / a
        x_feas = np.logical_and(lb < x, x < ub)
        a_pos = a > 0
        a_neg = a < 0

        x_lb = x == lb
        x_ub = x == ub

        min_mask = np.logical_or.reduce((x_feas,
                                        np.logical_and(x_lb,
                                                        a_pos),
                                        np.logical_and(x_ub,
                                                        a_neg)))
        max_mask = np.logical_or.reduce((x_feas,
                                        np.logical_and(x_lb,
                                                        a_neg),
                                        np.logical_and(x_ub,
                                                        a_pos)))


        if np.sum(max_mask) > 0:
            tmp = F.copy()
            tmp[np.logical_not(max_mask)] = -np.inf
            maxI = np.argmax(tmp)
            maxF = tmp[maxI]

        if np.sum(min_mask) > 0:
            tmp = F.copy()
            tmp[np.logical_not(min_mask)] = np.inf
            minI = np.argmin(tmp)
            minF = tmp[minI]

        # Check KKT conditions
        if (maxF - minF <= tolKKT):
            finished = True
            break

        # SMO update the most violating pair
        tau_lb_u = (lb[minI] - x[minI]) * a[minI]
        tau_ub_u = (ub[minI] - x[minI]) * a[minI]
        if (a[minI] <= 0):
            (tau_lb_u, tau_ub_u) = (tau_ub_u, tau_lb_u)

        tau_lb_v = (x[maxI] - lb[maxI]) * a[maxI]
        tau_ub_v = (x[maxI] - ub[maxI]) * a[maxI]
        if (a[maxI] > 0):
            (tau_lb_v, tau_ub_v) = (tau_ub_v, tau_lb_v)

        tau = (nabla[maxI] / a[maxI] - nabla[minI] / a[minI]) / \
              (H[minI, minI] / a[minI] ** 2 + H[maxI, maxI] / a[maxI] ** 2 - 2 * H[maxI, minI] / (a[minI] * a[maxI]))

        tau = min(max(tau, tau_lb_u, tau_lb_v), tau_ub_u, tau_ub_v)
        if (tau == 0):
            # Converged on a non-optimal solution
            break

        x[minI] += tau / a[minI]
        x[maxI] -= tau / a[maxI]

        nabla += H[:, minI] * tau / a[minI]
        nabla -= H[:, maxI] * tau / a[maxI]

        t += 1
        # Print iter info
        if (verb > 0 and t % verb == 0):
            obj = 0.5 * np.sum(x * nabla + x * f)
            print "t=%d, KKTviol=%f, tau=%f, tau_lb=%f, tau_ub=%f, Q_P=%f" % \
                  (t, maxF - minF, tau, max(tau_lb_u, tau_lb_v), min(tau_ub_u, tau_ub_v), obj)
            # raw_input()

    fval = 0.5 * np.dot(x, nabla + f)
    return x, fval, t, finished

def plot_pts(X, y):
    ''' Plots 2D points from two classes

    Args:
    - X (2xN np.array) - input data
    - y (1xN, np.array containing -1,+1) - class labels -1 / +1'''
    y = np.squeeze(y)
    pts_A = X[:, y > 0]
    pts_B = X[:, y < 0]

    plt.scatter(pts_A[0, :], pts_A[1, :])
    plt.scatter(pts_B[0, :], pts_B[1, :])

def plot_boundary(ax, model):
    ''' Plots 2-class linear decision boundary

    Args:
    - ax (matplotlib Axes) - axes to draw onto
    - model - dictionary returned by my_kernel_svm'''
    y_lim = ax.get_ylim()
    x_lim = ax.get_xlim()

    xs = np.linspace(x_lim[0], x_lim[1], 400)
    ys = np.linspace(y_lim[0], y_lim[1], 400)

    xs, ys = np.meshgrid(xs, ys, indexing='ij')
    X = np.vstack((xs.flatten(), ys.flatten()))

    z = model['fun'](X, model).reshape(xs.shape)
    plt.contour(xs, ys, z, [0])

def crossval(num_data, num_folds):
    """
    itrn, itst = crossval(num_data, num_folds)

    Partitions data for cross-validation.

    This function randomly partitions data into the training
    and testing parts. The number of partitioning is determined
    by the num_folds. If num_folds==1 then makes only one random
    partitioning of data into training and testing in ratio 50:50.

    :param num_data:    number of data (scalar, integer)
    :param num_folds:   number of folders (scalar, integer)
    :return:            itrn - LIST of training folds, itst - LIST of testing folds
                        itrn[i] indices of training data of i-th folder <1 x n> np array
                        itst[i] indices of testing data of i-th folder <1 x n> np array
    """
    if num_folds < 2:
        num_folds = 2

    inx = np.expand_dims(np.random.permutation(num_data), 0)

    itrn = []
    itst = []

    num_column = np.int32(np.ceil(np.float64(num_data) / num_folds))

    for idx in range(num_folds):
        tst_idx = range((idx * num_column), np.min([num_data, ((idx + 1) * num_column)]))
        trn_idx = [i for i in list(range(num_data)) if i not in tst_idx]
        itst.append(inx[:, tst_idx])
        itrn.append(inx[:, trn_idx])
    return itrn, itst

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

def show_classification(test_images, labels, letters):
    """
    show_classification(test_images, labels, letters)

    create montages of images according to estimated labels

    :param test_images:     shape h x w x n
    :param labels:          shape 1 x n
    :param letters:         string with letters, e.g. 'CN'
    """
    for i in range(len(letters)):
        imgs = test_images[:,:,labels[0]==i]
        subfig = plt.subplot(1,len(letters),i+1)
        montage(imgs)
        plt.title(letters[i])

def show_mnist_classification(X, y):
    """
    show_mnist_classification(X, y)

    create montages of mnist images according to estimated labels

    :param X:               < (28*28) x n > np.array of MNIST measurements
    :param y:               < 1 x n > np.array of labels (-1 represents 0, 1 represents 1)
    """
    y = np.squeeze(y)
    n_images = X.shape[1]
    images = np.zeros((28, 28, n_images));
    for i in range(n_images):
        images[:, :, i] = np.reshape(X[:, i], [28, 28]);

    plt.figure();
    plt.subplot(1, 2, 1)
    montage(images[:,:, y == -1]);
    plt.title('class 0');

    plt.subplot(1, 2, 2)
    montage(images[:,:, y == 1]);
    plt.title('class 1');
