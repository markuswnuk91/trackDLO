from builtins import super
from inspect import Parameter
import os
from re import S
import sys
import numpy as np
from scipy.linalg import schur
import numbers

try:
    sys.path.append(os.getcwd().replace("/src/tracking/spr", ""))
    from src.tracking.registration import NonRigidRegistration
    from src.tracking.utils.utils import (
        gaussian_kernel,
        initialize_sigma2,
        sqdistance_matrix,
    )
except:
    print("Imports for SPR failed.")
    raise


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


def mlle(X, k, d):
    """Construct locally linear embeddings to reconstuct it on a lower dimesional manifold.
    Implementation according to:
    Jing Wang and Zhenyue Zhang, Nonlinear Embedding Preserving Multiple Local-linearities, Patten Recognition, Vol.43, pp.1257-1268, 2010

    Args:
        X (np.array)= N x D matrix of points to be reduced
        k (int): number of nearest neighbors to be used for reconsturction of local embeddings
        d (int): disired dimension of the output data

    Returns:
        Phi:
        Y:
    """

    (N, D) = X.shape
    tol = 1e-3  # regularization parameter
    roh = np.zeros((N))
    # find Neighborhood
    (J, _) = knn(X, X, k + 1)  # one more because knn includes each point itself.

    for i in range(N):
        # step 1.1
        xi = X[i, :]
        Ji = J[i, 1:]  # neighbrhood of xi without xi itself.

        # step 1.2
        Gi = X[Ji, :] - xi
        C = Gi @ Gi.transpose()
        C_tilde = C + np.eye(k, k) * tol * np.trace(C)
        wiOpt = np.linalg.solve(C_tilde, np.ones((k, 1)))
        wiOpt = wiOpt / np.sum(wiOpt)  # regularization as suggested at p.3

        # step 1.3
        [S, V] = schur(C, output="real")
        ei = np.sort(np.diag(S))
        ei = ei[::-1]
        ei[ei <= np.finfo(float).eps] = 0
        JIi = np.argsort(np.diag(S))
        JIi = JIi[::-1]
        roh[i] = np.sum(ei[d:k]) / np.sum(ei[0:d])
    return 0


class StructurePreservedRegistration(NonRigidRegistration):
    """
    Implementation of the Structure Preserved Registration (SPR) according to
    the paper:
    Tang, T. and Tomizuka, M. (2022); "Track deformable objects from point clouds with structure preserved registration", The International Journal of Robotics Research, 41(6), pp. 599–614. doi: 10.1177/0278364919841431.
    Based on their provided Matlab implementation:
    https://github.com/thomastangucb/SPR

    Attributes
    ----------
    tauFactor: float (positive)
        Regularization factor for the Local Regularization (MLLE).
        A higher tauFactor enforces stronger local structure perservation between points.

    lambdaFactor: float (positive)
        Regularization factor for the Global Regularization (CPD).
        A higher lambdaFactor enforces stronger global coupling (coherent movement)between points.

    beta: float(positive)
        Width of the Gaussian kernel for global regularization.
        A higher beta enforces a stronger coherence in the movement of the points, the points behave "more stiff".

    sigma2: float (positive)
        Variance of the Gaussian mixture model

    diff: float (positive)
        The absolute normalized difference between the current and previous objective function values.

    L: float
        The value of the objective function.

    P: numpy array
        MxN array of probabilities.
        P[m, n] represents the probability that the m-th source point
        corresponds to the n-th target point.

    mu: float (between 0 and 1)
        Contribution of the uniform distribution to account for outliers.
        Valid values span 0 (inclusive) and 1 (exclusive).
    """

    def __init__(
        self,
        tauFactor=None,
        lambdaFactor=None,
        sigma2=None,
        beta=None,
        mu=None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        if tauFactor is not None and (
            not isinstance(tauFactor, numbers.Number) or tauFactor <= 0
        ):
            raise ValueError(
                "Expected a positive value for regularization parameter tau. Instead got: {}".format(
                    tauFactor
                )
            )

        if lambdaFactor is not None and (
            not isinstance(lambdaFactor, numbers.Number) or lambdaFactor <= 0
        ):
            raise ValueError(
                "Expected a positive value for regularization parameter tau. Instead got: {}".format(
                    lambdaFactor
                )
            )

        if beta is not None and (not isinstance(beta, numbers.Number) or beta <= 0):
            raise ValueError(
                "Expected a positive value for the width of the coherent Gaussian kernel. Instead got: {}".format(
                    beta
                )
            )

        if sigma2 is not None and (
            not isinstance(sigma2, numbers.Number) or sigma2 <= 0
        ):
            raise ValueError(
                "Expected a positive value for sigma2 instead got: {}".format(sigma2)
            )

        if mu is not None and (not isinstance(mu, numbers.Number) or mu < 0 or mu >= 1):
            raise ValueError(
                "Expected a value between 0 (inclusive) and 1 (exclusive) for mu instead got: {}".format(
                    mu
                )
            )

        self.tauFactor = 3 if tauFactor is None else tauFactor
        self.lambdaFactor = 3 if lambdaFactor is None else lambdaFactor
        self.beta = 2 if beta is None else beta
        self.sigma2 = initialize_sigma2(self.X, self.Y) if sigma2 is None else sigma2
        self.mu = 0.0 if mu is None else mu
        self.diff = np.inf
        self.L = np.inf

        self.P = np.zeros((self.N, self.M))
        self.Pt1 = np.zeros((self.M,))
        self.P1 = np.zeros((self.N,))
        self.PY = np.zeros((self.N, self.D))
        self.PY = np.zeros((self.N, self.D))
        self.W = np.zeros((self.N, self.D))
        self.G = gaussian_kernel(self.X, self.beta)

        def isConverged(self):
            """
            Checks if change of cost function is below the defined tolerance
            """
            return self.diff < self.tolerance

        def estimateCorrespondance(self):
            """
            E-step: Compute the expectation step  of the EM algorithm.
            """
            P = np.sum((self.Y[None, :, :] - self.T[:, None, :]) ** 2, axis=2)

            c = (2 * np.pi * self.sigma2) ** (self.D / 2)
            c = c * self.mu / (1 - self.mu)
            c = c * self.N / self.M

            P = np.exp(-P / (2 * self.sigma2))
            den = np.sum(P, axis=0)
            den = np.tile(den, (self.N, 1))
            den[den == 0] = np.finfo(float).eps
            den += c

            self.P = np.divide(P, den)
            self.Pt1 = np.sum(self.P, axis=0)
            self.P1 = np.sum(self.P, axis=1)
            self.Np = np.sum(self.P1)
            self.PY = np.matmul(self.P, self.Y)
