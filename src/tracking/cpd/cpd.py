from builtins import super
import os
import sys
import numpy as np
import numbers

try:
    sys.path.append(os.getcwd().replace("/src/tracking/cpd", ""))
    from src.tracking.registration import NonRigidRegistration
    from src.utils.utils import gaussian_kernel, initialize_sigma2
except:
    print("Imports for CPD failed.")
    raise


def low_rank_eigen(G, num_eig):
    """
    Calculate num_eig eigenvectors and eigenvalues of gaussian matrix G.
    Enables lower dimensional solving.
    Attributes
    ----------
    G: numpy array
        Gaussian kernel matrix.

    num_eig: int
        Number of eigenvectors to use in lowrank calculation.

    Returns
    -------
    Q: numpy array
        D x num_eig array of eigenvectors.

    S: numpy array
        num_eig array of eigenvalues.
    """
    S, Q = np.linalg.eigh(G)
    eig_indices = list(np.argsort(np.abs(S))[::-1][:num_eig])
    Q = Q[:, eig_indices]  # eigenvectors
    S = S[eig_indices]  # eigenvalues.
    return Q, S


class CoherentPointDrift(NonRigidRegistration):
    """
    Implementation of the Coherent Point Drift Algorithm (CPD) according to:
    https://github.com/siavashk/pycpd/

    Attributes
    ----------
    alpha: float (positive)
        Represents the trade-off between the goodness of maximum likelihood fit and regularization.

    beta: float(positive)
        Width of the Gaussian kernel.

    sigma2: float (positive)
        Initial variance of the Gaussian mixture model.

    mu: float (between 0 and 1)
        Contribution of the uniform distribution to account for outliers.
        Valid values span 0 (inclusive) and 1 (exclusive).

    diff: float (positive)
        The absolute difference between the current and previous objective function values.

    P: numpy array
        MxN array of probabilities.
        P[m, n] represents the probability that the m-th source point
        corresponds to the n-th target point.

    Pt1: numpy array
        Nx1 column array.
        Multiplication result between the transpose of P and a column vector of all 1s.

    P1: numpy array
        Mx1 column array.
        Multiplication result between P and a column vector of all 1s.

    Np: float (positive)
        The sum of all elements in P.

    low_rank: bool
        Whether to use low rank approximation.

    num_eig: int
        Number of eigenvectors to use in lowrank calculation.

    q: float
        Objective function CPD aims to minimize (negative log-likelyhood)
    """

    def __init__(
        self, alpha=None, beta=None, low_rank=False, num_eig=100, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        if alpha is not None and (not isinstance(alpha, numbers.Number) or alpha <= 0):
            raise ValueError(
                "Expected a positive value for regularization parameter alpha. Instead got: {}".format(
                    alpha
                )
            )

        if beta is not None and (not isinstance(beta, numbers.Number) or beta <= 0):
            raise ValueError(
                "Expected a positive value for the width of the coherent Gaussian kernel. Instead got: {}".format(
                    beta
                )
            )
        self.initializeParameters(alpha, beta, low_rank, num_eig)

    def initializeParameters(self, alpha=None, beta=None, low_rank=False, num_eig=100):
        self.alpha = 2 if alpha is None else alpha
        self.beta = 2 if beta is None else beta
        self.diff = np.inf
        self.L = np.inf
        self.W = np.zeros((self.N, self.D))
        self.G = gaussian_kernel(self.X, self.beta)
        self.low_rank = low_rank
        self.num_eig = num_eig
        if self.low_rank is True:
            self.Q, self.S = low_rank_eigen(self.G, self.num_eig)
            self.inv_S = np.diag(1.0 / self.S)
            self.S = np.diag(self.S)
            self.E = 0.0
        self.initializeCorrespondances()

    def initializeCorrespondances(self):
        self.P = np.zeros((self.N, self.M))
        self.Pden = np.zeros((self.M))
        self.Pt1 = np.zeros((self.M,))
        self.P1 = np.zeros((self.N,))
        self.PY = np.zeros((self.N, self.D))
        self.Np = 0

    def reinitializeParameters(self):
        self.initializeCorrespondances()
        self.estimateCorrespondance()
        self.update_variance()

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
        den[den == 0] = np.finfo(float).eps  # makes sure we do not divide by zero
        den += c

        self.Pden = den[0, :]
        self.P = np.divide(P, den)
        self.Pt1 = np.sum(self.P, axis=0)
        self.P1 = np.sum(self.P, axis=1)
        self.Np = np.sum(self.P1)
        self.PY = np.matmul(self.P, self.Y)

    def updateParameters(self):
        """
        M-step: Calculate a new estimate of the deformable transformation.
        See Eq. 22 of https://arxiv.org/pdf/0905.2635.pdf.
        """
        if self.low_rank is False:
            A = np.dot(np.diag(self.P1), self.G) + self.alpha * self.sigma2 * np.eye(
                self.N
            )
            B = self.PY - np.dot(np.diag(self.P1), self.X)
            self.W = np.linalg.solve(A, B)

        elif self.low_rank is True:
            # Matlab code equivalent can be found here:
            # https://github.com/markeroon/matlab-computer-vision-routines/tree/master/third_party/CoherentPointDrift
            dP = np.diag(self.P1)
            dPQ = np.matmul(dP, self.Q)
            F = self.PY - np.matmul(dP, self.X)

            self.W = (
                1
                / (self.alpha * self.sigma2)
                * (
                    F
                    - np.matmul(
                        dPQ,
                        (
                            np.linalg.solve(
                                (
                                    self.alpha * self.sigma2 * self.inv_S
                                    + np.matmul(self.Q.T, dPQ)
                                ),
                                (np.matmul(self.Q.T, F)),
                            )
                        ),
                    )
                )
            )
            QtW = np.matmul(self.Q.T, self.W)
            self.E = self.E + self.alpha / 2 * np.trace(
                np.matmul(QtW.T, np.matmul(self.S, QtW))
            )
        self.computeTargets()
        self.update_variance()

    def computeTargets(self, X=None):
        """
        Update the targets using the new estimate of the parameters.
        Attributes
        ----------
        X: numpy array, optional
            Array of points to transform - use to predict on new set of points.
            Best for predicting on new points not used to run initial registration.
                If None, self.X used.

        Returns
        -------
        If X is None, returns None.
        Otherwise, returns the transformed X.

        """
        if X is not None:
            G = gaussian_kernel(X=X, beta=self.beta, Y=self.X)
            return X + np.dot(G, self.W)
        else:
            if self.low_rank is False:
                self.T = self.X + np.dot(self.G, self.W)

            elif self.low_rank is True:
                self.T = self.X + np.matmul(
                    self.Q, np.matmul(self.S, np.matmul(self.Q.T, self.W))
                )
                return

    def update_variance(self):
        """
        Update the variance of the mixture model using the new estimate of the deformable transformation.
        See the update rule for sigma2 in Eq. 23 of of https://arxiv.org/pdf/0905.2635.pdf.
        """

        # The original CPD paper does not explicitly calculate the objective functional.
        # This functional includes terms from both the negative log-likelihood and
        # the Gaussian kernel used for regularization.
        Lold = self.L
        self.L = (
            np.sum(-np.log(self.Pden))
            - self.D * self.M * np.log(self.sigma2) / 2
            + self.alpha / 2 * np.trace(np.transpose(self.W) @ self.G @ self.W)
        )
        self.diff = np.abs((self.L - Lold) / self.L)

        # Optionally we could use the difference between the current and previous
        # estimate of the variance as a proxy to test for convergence.
        # qprev = self.sigma2 #comment in to use variance to test convergence

        yPy = np.dot(
            np.transpose(self.Pt1), np.sum(np.multiply(self.Y, self.Y), axis=1)
        )
        xPx = np.dot(np.transpose(self.P1), np.sum(np.multiply(self.T, self.T), axis=1))
        trPXY = np.sum(np.multiply(self.T, self.PY))

        self.sigma2 = (xPx - 2 * trPXY + yPy) / (self.Np * self.D)

        if self.sigma2 <= 0:
            self.sigma2 = self.tolerance / 10

        # self.diff = np.abs(self.sigma2 - qprev) # comment in to use variance to test convergence

    def getParameters(self):
        """
        Return the current estimate of the deformable transformation parameters.
        Returns
        -------
        self.G: numpy array
            Gaussian kernel matrix.
        self.W: numpy array
            Deformable transformation matrix.
        """
        return self.G, self.W
