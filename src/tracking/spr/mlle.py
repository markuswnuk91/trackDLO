import os
import sys
import numpy as np
from scipy.linalg import schur
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
import numbers
from warnings import warn

try:
    sys.path.append(os.getcwd().replace("/src/tracking/spr", ""))
    from src.tracking.registration import NonRigidRegistration
    from src.utils.utils import knn
except:
    print("Imports for MLLE failed.")
    raise


class mlle:
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

    def __init__(self, X, k=None, d=None):
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
        self.k = k
        self.k = 5 if k is None else k
        self.d = 2 if d is None else d
        self.tol = 1e-3  # regularization parameter

    def _computePhi(self):
        (N, D) = self.X.shape
        roh = np.zeros((N))
        eigenValues = []
        eigenVectors = []
        WOpti = []

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
            Gi = self.X[Ji, :] - xi
            C = Gi @ Gi.transpose()
            C_tilde = C + np.eye(self.k, self.k) * self.tol * np.trace(C)
            wiOpt = np.linalg.solve(C_tilde, np.ones((self.k, 1)))
            wiOpt = wiOpt / np.sum(wiOpt)  # regularization as suggested at p.3

            # step 1.3
            [S, V] = schur(C, output="real")
            ei = np.sort(np.diag(S))
            ei = ei[::-1]
            ei[ei <= np.finfo(float).eps] = 0
            JIi = np.argsort(np.diag(S))
            JIi = JIi[::-1]
            roh[i] = np.sum(ei[self.d : self.k]) / np.sum(ei[0 : self.d])

            # save eigenvalues and vectors for next steps
            eigenValues.append(ei)
            eigenVectors.append(V[:, JIi])
            WOpti.append(wiOpt)

        # step 2
        rohSorted = np.sort(roh)
        eta = rohSorted[int(np.ceil(N / 2))]
        s = np.zeros(N)
        for i in range(N):
            l = self.k - self.d
            lambdas = eigenValues[i]
            while (
                np.sum(lambdas[self.k - l :]) / np.sum(lambdas[: self.k - l]) > eta
                and l > 1
            ):
                l = l - 1

            s[i] = l

        # step 3
        self.Phi = np.zeros((N, N))
        for i in range(N):
            Ji = J[i, 1:]
            Vi = eigenVectors[i]
            Vhat = Vi[:, int(self.k - s[i]) :]
            vi = np.sum(Vhat, 0)  # equivalent to Vhat.transpose() @ np.ones(k, 1)
            alphai = np.linalg.norm(vi) / np.sqrt(s[i])
            ui = np.ones((int(s[i]))) * alphai - vi
            uiNorm = np.linalg.norm(ui)
            if uiNorm > 1e-5:
                ui = ui / uiNorm
            else:
                ui = np.zeros((int(s[i]), 1))
            Hi = np.eye(int(s[i])) - 2 * np.outer(ui, ui)
            Wi = (1 - alphai) ** 2 * WOpti[i] @ np.ones((1, int(s[i]))) + (
                2 - alphai
            ) * Vhat @ Hi

            # build Phi
            self.Phi[i, i] = self.Phi[i, i] + s[i]
            self.Phi[np.ix_(Ji, Ji)] = self.Phi[np.ix_(Ji, Ji)] + Wi @ Wi.transpose()
            self.Phi[Ji, i] = self.Phi[Ji, i] - np.sum(Wi, 1)
            self.Phi[i, Ji] = self.Phi[Ji, i]
        return self.Phi

    def getAlignmentMatrix(self):
        self._computePhi()
        return self.Phi

    def solve(self):
        # step 4
        Phi_ = self._computePhi()
        if self.Phi.shape[0] > 200 and self.d + 1 < 10:
            solver = "sparse"
        else:
            solver = "dense"

        if solver == "sparse":
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
                    "case, solver='dense' is recommended."
                ) from e
            # index = np.argsort(np.abs(eiVals))
            # Y = eiVecs[:, index[1:]]
            Y = eiVecs[:, 1:]
        elif solver == "dense":
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
            raise ValueError("Unrecognized solver '%s'" % solver)

        return Y
