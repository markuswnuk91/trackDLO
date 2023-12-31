import os
import sys
import numpy as np
from scipy.spatial import distance_matrix
from scipy.linalg import schur
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
import numbers
from warnings import warn

try:
    sys.path.append(os.getcwd().replace("/src/localization/downsampling/mlle", ""))
    from src.utils.utils import knn
except:
    print("Imports for MLLE failed.")
    raise


class Lle:
    """Class for constructing locally linear embeddings to reconstuct it on a lower dimesional manifold.
    Implementation according to:
    Jing Wang and Zhenyue Zhang, Nonlinear Embedding Preserving Multiple Local-linearities, Patten Recognition, Vol.43, pp.1257-1268, 2010

    Attributes
    ----------
    Input:
        X (np.array): N x D matrix of points to be reduced
        k (int): number of nearest neighbors to be used for reconsturction of local embeddings
        d (int): desired dimension of the output data

    Output:

        Phi (np.array): NxN alignment matrix
        Y (np.array): Nxd Data points given in the reduced dimension

    Use:
        _computePhi: computes alignment matrix Phi
        getAlignmentMatrix : returns algignment matrix phi
        solve: returns Points in reduced coordinates Y
    """

    def __init__(
        self,
        X,
        k=None,
        d=None,
        tol=None,
        solverType=None,
        mapping=None,
        exponent=None,
        sigma=None,
        *args,
        **kwargs
    ):
        if type(X) is not np.ndarray or X.ndim != 2:
            raise ValueError("The target point cloud (X) must be at a 2D numpy array.")
        if X.shape[0] < X.shape[1]:
            raise ValueError(
                "The dimensionality is larger than the number of points. Possibly the wrong orientation of X."
            )

        if X.shape[0] < d:
            raise ValueError(
                "The dimensionality of the given dataset is already smaller than the desired dimensonality."
            )

        if k is not None and (not isinstance(k, numbers.Number) or k < 0):
            raise ValueError(
                "Expected a positive integer for number of nearest neighbors instead got: {}".format(
                    k
                )
            )
        elif isinstance(k, numbers.Number) and not isinstance(k, int):
            warn(
                "Received a non-integer value for max_iterations: {}. Casting to integer.".format(
                    k
                )
            )
            k = int(k)

        if d is not None and (not isinstance(d, numbers.Number) or d < 0):
            raise ValueError(
                "Expected a positive integer for number of dimensions instead got: {}".format(
                    d
                )
            )
        elif isinstance(d, numbers.Number) and not isinstance(d, int):
            warn(
                "Received a non-integer value for max_iterations: {}. Casting to integer.".format(
                    d
                )
            )
            d = int(d)
        self.X = X
        self.k = 5 if k is None else k
        self.d = 2 if d is None else d
        self.tol = 1e-3 if tol is None else tol  # regularization parameter
        self.mapping = "linear" if mapping is None else mapping
        self.sigma = 1 if sigma is None else sigma
        self.exponent = 1 if exponent is None else exponent
        self.solverType = "dense" if solverType is None else solverType

    def _computePhi(self):
        (N, D) = self.X.shape
        W = np.zeros((N, N))

        # find Neighborhood
        (J, _) = knn(
            self.X, self.X, self.k + 1
        )  # one more because knn includes each point itself.

        # step 1
        for i in range(N):
            # step 1.1
            xi = self.X[i, :]
            Ji = J[i, 1:]  # neighbrhood of xi without xi itself.

            # step 1.2
            # Gi = (self.X[Ji, :] - xi) * (
            #     1
            #     - np.exp(
            #         -np.expand_dims(np.linalg.norm(self.X[Ji, :] - xi, axis=1), axis=-1)
            #         ** self.exponent
            #         / (2 * self.sigma**2)
            #     )
            # )
            Gi = self.X[Ji, :] - xi
            C = Gi @ Gi.transpose()
            # if self.mapping == "linear":
            #     C = Gi @ Gi.transpose()
            # elif self.mapping == "power":
            #     C = Gi @ Gi.transpose() ** self.exponent * (Gi @ Gi.transpose())
            # elif self.mapping == "exponential":S
            #     Gi_hat = np.expand_dims(kernelMatrix[Ji, i], axis=-1) * Gi
            #     C = Gi_hat @ Gi_hat.transpose()
            C_tilde = C + np.eye(self.k, self.k) * self.tol * np.trace(C)
            wi = np.linalg.solve(C_tilde, np.ones((self.k, 1)))
            wi = wi / np.sum(wi)  # regularization as suggested at p.3
            if self.mapping == "linear":
                pass
            if self.mapping == "power":
                wi = wi * wi ** (self.exponent - 1)
            elif self.mapping == "exponential":
                wi = wi * np.exp(-(wi**2) / (2 * self.sigma**2))
            W[i, Ji] = wi.flatten()

        M = np.eye(W.shape[0], W.shape[1]) - W
        self.Phi = M.T @ M

        return self.Phi

    def getAlignmentMatrix(self):
        self._computePhi()
        return self.Phi

    def solve(self):
        kernelMatrix = np.exp(
            -(distance_matrix(self.X, self.X, p=self.exponent)) / (2 * self.sigma**2)
        )
        # step 4
        Phi_ = self._computePhi()
        if self.solverType == "sparse":
            try:
                tol = 1e-6
                max_iter = 100
                # eiVals, eiVecs = eigsh(
                #     self.Phi, self.d + 1, maxiter=max_iter, which="SM"
                # )
                eiVals, eiVecs = eigsh(
                    self.Phi, self.d + 1, maxiter=max_iter, sigma=0.0, tol=tol
                )
            except RuntimeError as e:
                raise ValueError(
                    "Error in determining null-space with sparse solver. Error message: "
                    "'%s'. Note that solving the eigenvalue problem can fail when the "
                    "weight matrix is singular or otherwise ill-behaved. In that "
                    "case, solverType='dense' is recommended."
                ) from e
            # index = np.argsort(np.abs(eiVals))
            # Y = eiVecs[:, index[1:]]
            Y = eiVecs[:, 1:]
        elif self.solverType == "dense":
            try:
                eiVals, eiVecs = eigh(
                    Phi_, subset_by_index=[1, self.d], overwrite_a=True
                )
            except RuntimeError as e:
                raise ValueError(
                    "Error in determining null-space with dense solver."
                ) from e
            Y = eiVecs
        else:
            raise ValueError("Unrecognized solver '%s'" % self.solverType)
        return Y
