import numpy as np


def sqdistance_matrix(X, Y=None):
    """
    Calculate the squared distance matrix.
    Computing the squared distance is computationally easier, since the root is not required

    Attributes
    ----------
    X: numpy array
        NxD array of points.

    Y: numpy array, optional
        MxD array of secondary points to calculate
        distance with.

    Returns
    -------
    diff: numpy array
        Distance matrix.
            NxN if Y is None
            NxM if Y is not None
    """
    if Y is None:
        Y = X
    diff = X[:, None, :] - Y[None, :, :]
    diff = np.square(diff)
    diff = np.sum(diff, 2)
    return diff


def gaussian_kernel(X, beta, Y=None):
    """
    Calculate gaussian kernel matrix.
    Attributes
    ----------
    X: numpy array
        NxD array of points for creating gaussian.

    beta: float
        Width of the Gaussian kernel.

    Y: numpy array, optional
        MxD array of secondary points to calculate
        kernel with. Used if predicting on points
        not used to train.

    Returns
    -------
    K: numpy array
        Gaussian kernel matrix.
            NxN if Y is None
            NxM if Y is not None
    """
    if Y is None:
        Y = X
    diff = X[:, None, :] - Y[None, :, :]
    diff = np.square(diff)
    diff = np.sum(diff, 2)
    return np.exp(-diff / (2 * beta**2))


def initialize_sigma2(X, Y):
    """
    Initialize the variance (sigma2).
    Attributes
    ----------
    Y: numpy array
        NxD array of points for target.

    X: numpy array
        MxD array of points for source.

    Returns
    -------
    sigma2: float
        Initial variance.
    """
    (N, D) = X.shape
    (M, _) = Y.shape
    diff = X[None, :, :] - Y[:, None, :]
    err = diff**2
    return np.sum(err) / (D * M * N)


def knn(X, Y, k):
    """
    Finds the k nearest neighbors of Y in X. (Brute Force Implementation)
    Input:
    X = N x D matrix. N=rows and D=dimensions/features
    Y = M x D matrix. M=rows and D=dimensions/features
    k = number of nearest neighbors to be found

    Output:
    indices = Nxk matrix of indices of k nearest neighbors of Y to X
    dists = distances between X/Y points. Size of N x M
    """
    sqdistances = sqdistance_matrix(X, Y)
    indices = np.argsort(sqdistances, 1)
    distances = np.sort(sqdistances, 1)
    distances = distances[:, 0:k] ** 0.5
    distances[
        distances <= np.finfo(float).eps
    ] = 0  # makes sure we do not calculate negative values
    return indices[:, 0:k], distances
