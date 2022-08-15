from builtins import super
import os
import sys
import numpy as np
import numbers

try:
    sys.path.append(os.getcwd().replace("/src/tracking/cpd", ""))
    from src.tracking.registration import NonRigidRegistration
except:
    print("Imports for CPD failed.")
    raise


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
    diff = X[:, None, :] - Y[None, :,  :]
    diff = np.square(diff)
    diff = np.sum(diff, 2)
    return np.exp(-diff / (2 * beta**2))

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

def initialize_sigma2(X, Y):
    """
    Initialize the variance (sigma2).
    Attributes
    ----------
    X: numpy array
        NxD array of points for target.
    
    Y: numpy array
        MxD array of points for source.
    
    Returns
    -------
    sigma2: float
        Initial variance.
    """
    (N, D) = X.shape
    (M, _) = Y.shape
    diff = X[None, :, :] - Y[:, None, :]
    err = diff ** 2
    return np.sum(err) / (D * M * N)

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
    """

    def __init__(self, alpha=None, beta=None, sigma2=None, w=None, low_rank=False, num_eig=100, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if alpha is not None and (not isinstance(alpha, numbers.Number) or alpha <= 0):
            raise ValueError(
                "Expected a positive value for regularization parameter alpha. Instead got: {}".format(alpha))
             
        if beta is not None and (not isinstance(beta, numbers.Number) or beta <= 0):
            raise ValueError(
                "Expected a positive value for the width of the coherent Gaussian kerenl. Instead got: {}".format(beta))

        if sigma2 is not None and (not isinstance(sigma2, numbers.Number) or sigma2 <= 0):
            raise ValueError(
                "Expected a positive value for sigma2 instead got: {}".format(sigma2))
        
        if w is not None and (not isinstance(w, numbers.Number) or w < 0 or w >= 1):
            raise ValueError(
                "Expected a value between 0 (inclusive) and 1 (exclusive) for w instead got: {}".format(w))
           
        self.alpha = 2 if alpha is None else alpha
        self.beta = 2 if beta is None else beta
        self.sigma2 = initialize_sigma2(self.X, self.Y) if sigma2 is None else sigma2
        self.w = 0.0 if w is None else w
        self.diff = np.inf
        self.q = np.inf
        self.P = np.zeros((self.M, self.N))
        self.Pt1 = np.zeros((self.N, ))
        self.P1 = np.zeros((self.M, ))
        self.PX = np.zeros((self.M, self.D))
        self.Np = 0
        self.W = np.zeros((self.M, self.D))
        self.G = gaussian_kernel(self.Y, self.beta)
        self.low_rank = low_rank
        self.num_eig = num_eig
        if self.low_rank is True:
            self.Q, self.S = low_rank_eigen(self.G, self.num_eig)
            self.inv_S = np.diag(1./self.S)
            self.S = np.diag(self.S)
            self.E = 0.

    def isConverged(self):
        """
        Checks if change of cost function is below the defined tolerance
        """
        return self.diff < self.tolerance

    def estimateCorrespondance(self):
        """
        Compute the expectation step of the EM algorithm.
        """
        P = np.sum((self.X[None, :, :] - self.T[:, None, :]) ** 2, axis=2)

        c = (2 * np.pi * self.sigma2) ** (self.D / 2)
        c = c * self.w / (1 - self.w)
        c = c * self.M / self.N

        P = np.exp(-P / (2 * self.sigma2))
        den = np.sum(P, axis=0)
        den = np.tile(den, (self.M, 1))
        den[den == 0] = np.finfo(float).eps
        den += c

        self.P = np.divide(P, den)
        self.Pt1 = np.sum(self.P, axis=0)
        self.P1 = np.sum(self.P, axis=1)
        self.Np = np.sum(self.P1)
        self.PX = np.matmul(self.P, self.X)

    def computeTargets(self, Y=None):
        """
        Update the targets using the new estimate of the parameters.
        Attributes
        ----------
        Y: numpy array, optional
            Array of points to transform - use to predict on new set of points.
            Best for predicting on new points not used to run initial registration.
                If None, self.Y used.
        
        Returns
        -------
        If Y is None, returns None.
        Otherwise, returns the transformed Y.
                
        """
        if Y is not None:
            G = gaussian_kernel(X=Y, beta=self.beta, Y=self.Y)
            return Y + np.dot(G, self.W)
        else:
            if self.low_rank is False:
                self.T = self.Y + np.dot(self.G, self.W)

            elif self.low_rank is True:
                self.T = self.Y + np.matmul(self.Q, np.matmul(self.S, np.matmul(self.Q.T, self.W)))
                return

    def updateParameters(self):
        """
        Calculate a new estimate of the deformable transformation.
        See Eq. 22 of https://arxiv.org/pdf/0905.2635.pdf.
        """
        if self.low_rank is False:
            A = np.dot(np.diag(self.P1), self.G) + \
                self.alpha * self.sigma2 * np.eye(self.M)
            B = self.PX - np.dot(np.diag(self.P1), self.Y)
            self.W = np.linalg.solve(A, B)

        elif self.low_rank is True:
            # Matlab code equivalent can be found here:
            # https://github.com/markeroon/matlab-computer-vision-routines/tree/master/third_party/CoherentPointDrift
            dP = np.diag(self.P1)
            dPQ = np.matmul(dP, self.Q)
            F = self.PX - np.matmul(dP, self.Y)

            self.W = 1 / (self.alpha * self.sigma2) * (F - np.matmul(dPQ, (
                np.linalg.solve((self.alpha * self.sigma2 * self.inv_S + np.matmul(self.Q.T, dPQ)),
                                (np.matmul(self.Q.T, F))))))
            QtW = np.matmul(self.Q.T, self.W)
            self.E = self.E + self.alpha / 2 * np.trace(np.matmul(QtW.T, np.matmul(self.S, QtW)))
        
        self.update_variance()

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

    def update_variance(self):
        """
        Update the variance of the mixture model using the new estimate of the deformable transformation.
        See the update rule for sigma2 in Eq. 23 of of https://arxiv.org/pdf/0905.2635.pdf.
        """
        qprev = self.sigma2

        # The original CPD paper does not explicitly calculate the objective functional.
        # This functional will include terms from both the negative log-likelihood and
        # the Gaussian kernel used for regularization.
        self.q = np.inf

        xPx = np.dot(np.transpose(self.Pt1), np.sum(
            np.multiply(self.X, self.X), axis=1))
        yPy = np.dot(np.transpose(self.P1),  np.sum(
            np.multiply(self.T, self.T), axis=1))
        trPXY = np.sum(np.multiply(self.T, self.PX))

        self.sigma2 = (xPx - 2 * trPXY + yPy) / (self.Np * self.D)

        if self.sigma2 <= 0:
            self.sigma2 = self.tolerance / 10

        # Here we use the difference between the current and previous
        # estimate of the variance as a proxy to test for convergence.
        self.diff = np.abs(self.sigma2 - qprev)