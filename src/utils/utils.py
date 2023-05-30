import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import distance_matrix


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


def minimalSpanningTree(featureMatrix):
    """Returns the minimal spanning tree betwwen the nodes spanning the feature matrix
    Args:
        featureMatrix(np.array):
            feature Matrix containing the cost between all nodes

    Returns:
        symmetricAdjacencyMatrix(csgraph):
            symmetric adjacencyMatrix
    """
    if type(featureMatrix) is not np.ndarray or featureMatrix.ndim != 2:
        raise ValueError("feature matrix must be at a 2D numpy array.")

    if featureMatrix.shape[0] != featureMatrix.shape[1]:
        raise ValueError(
            "The feature matrix must be square and same have the same length as X. Instead got {}".format(
                featureMatrix.shape[0]
            )
        )
    adjacencyMatrix = minimum_spanning_tree(featureMatrix)
    symmetricAdjacencyMatrix = (
        adjacencyMatrix.toarray().astype(float)
        + adjacencyMatrix.toarray().astype(float).T
    )
    return symmetricAdjacencyMatrix


def calculateCorrespondance(X, Y):
    """calculates the correspondance between two point sets

    Args:
        X (np.array): point set of size NxD
        Y (np.array): point set of size MxD

    Returns:
        C (list of np.array): correspondances between X and Y, such that Y[C[i],:] are the points in Y corresponding to X[i]
    """
    C = []
    (N, D) = X.shape
    (M, _) = Y.shape
    distances = distance_matrix(X, Y)
    correspondingIndices = np.argmin(distances, axis=0)
    for i in range(0, N):
        C.append(np.where(correspondingIndices == i)[0])
    return C


def dampedPseudoInverse(J, dampingFactor):
    dim = J.shape[0]
    dampedPseudoInverse = J.T @ np.linalg.inv(
        J @ J.T + dampingFactor**2 * np.eye(dim)
    )
    return dampedPseudoInverse
