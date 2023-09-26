from builtins import super
import os
import sys
import numpy as np
import numbers

try:
    sys.path.append(os.getcwd().replace("/src/tracking/spr", ""))
    from src.tracking.registration import NonRigidRegistration
    from src.localization.downsampling.mlle.mlle import Mlle
    from src.utils.utils import (
        gaussian_kernel,
        initialize_sigma2,
    )
except:
    print("Imports for SPR failed.")
    raise


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
        The log-likelyhood of the dataset probability given the parameterization. SPR aims to update the parameters such that they maximize this value.

    P: numpy array
        MxN array of probabilities.
        P[m, n] represents the probability that the m-th source point
        corresponds to the n-th target point.

    mu: float (between 0 and 1)
        Contribution of the uniform distribution to account for outliers.
        Valid values span 0 (inclusive) and 1 (exclusive).

    knn: int (>=3)
        Number of nearest neighbors used for contructing the local regularization from Modified Locally Linear Embeddings (MLLE).

    tauAnnealing: float (between 0 and 1)
        Annealing factor for the local regularization (MLLE). The factor is reduced every iteration by the annealing factor. 1 is no annealing.

    lambdaAnnealing: float (between 0 and 1)
        Annealing factor for the global regularization (CPD). The factor is reduced every iteration by the annealing factor. 1 is no annealing.
    """

    def __init__(
        self,
        tauFactor=None,
        lambdaFactor=None,
        beta=None,
        knn=None,
        tauAnnealing=None,
        lambdaAnnealing=None,
        log=False,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        if tauFactor is not None and (
            not isinstance(tauFactor, numbers.Number) or tauFactor < 0
        ):
            raise ValueError(
                "Expected a positive value for regularization parameter tau. Instead got: {}".format(
                    tauFactor
                )
            )

        if lambdaFactor is not None and (
            not isinstance(lambdaFactor, numbers.Number) or lambdaFactor < 0
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

        if knn is not None and (not isinstance(knn, numbers.Number) or knn < 3):
            raise ValueError(
                "Expected at least 3 neighbors to reconstruct the local neighborhood instead got: {}".format(
                    knn
                )
            )

        if tauAnnealing is not None and (
            not isinstance(tauAnnealing, numbers.Number)
            or tauAnnealing <= 0
            or tauAnnealing > 1
        ):
            raise ValueError(
                "Expected a value between 0 and 1 for tauAnnealing instead got: {}".format(
                    tauAnnealing
                )
            )

        if lambdaAnnealing is not None and (
            not isinstance(lambdaAnnealing, numbers.Number)
            or lambdaAnnealing <= 0
            or lambdaAnnealing > 1
        ):
            raise ValueError(
                "Expected a value between 0 and 1 for lambdaAnnealing instead got: {}".format(
                    lambdaAnnealing
                )
            )
        self.initializeParameters(
            tauFactor=tauFactor,
            lambdaFactor=lambdaFactor,
            beta=beta,
            knn=knn,
            tauAnnealing=tauAnnealing,
            lambdaAnnealing=lambdaAnnealing,
        )

    def initializeParameters(
        self,
        tauFactor=None,
        lambdaFactor=None,
        beta=None,
        knn=None,
        tauAnnealing=None,
        lambdaAnnealing=None,
    ):
        self.tauFactor = 2 if tauFactor is None else tauFactor
        self.lambdaFactor = 2 if lambdaFactor is None else lambdaFactor
        self.beta = 2 if beta is None else beta
        self.knn = 7 if knn is None else knn
        self.tauAnnealing = 0.97 if tauAnnealing is None else tauAnnealing
        self.lambdaAnnealing = 0.97 if lambdaAnnealing is None else lambdaAnnealing
        self.diff = np.inf
        self.L = -np.inf

        self.W = np.zeros((self.N, self.D))
        self.G = gaussian_kernel(self.X, self.beta)
        self.Phi = Mlle(self.X, knn, 2).getAlignmentMatrix()

        if self.logging:
            self.log["W"] = [self.W]
            self.log["G"] = self.G
            self.log["sigma2"] = [self.sigma2]

    def initializeCorrespondances(self):
        self.P = np.zeros((self.N, self.M))
        self.Pden = np.zeros((self.M))
        self.Pt1 = np.zeros((self.M,))
        self.P1 = np.zeros((self.N,))
        self.Np = 0
        self.PY = np.zeros((self.N, self.D))

    def initializeWeights(self):
        self.W = np.zeros((self.N, self.D))

    def reinitializeParameters(self):
        self.initializeWeights()
        self.initializeCorrespondances()
        self.estimateCorrespondance()
        self.update_variance()

    def isConverged(self):
        """
        Checks if change of cost function is below the defined tolerance
        """
        return self.diff < self.tolerance

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
            self.T = self.X + np.dot(self.G, self.W)
            return

    def updateParameters(self):
        """
        M-step: Calculate a new parameters of the registration.
        """

        self.tauFactor = (self.tauAnnealing) ** (self.iteration) * self.tauFactor
        self.lambdaFactor = (self.lambdaAnnealing) ** (
            self.iteration
        ) * self.lambdaFactor

        dP1 = np.diag(self.P1)
        A = (
            np.dot(dP1, self.G)
            + self.lambdaFactor * self.sigma2 * np.eye(self.N)
            + self.tauFactor * self.sigma2 * np.dot(self.Phi, self.G)
        )
        B = (
            self.PY
            - np.dot(dP1, self.X)
            - self.tauFactor * self.sigma2 * np.dot(self.Phi, self.X)
        )
        self.W = np.linalg.solve(A, B)

        # set the new targets
        self.computeTargets()
        self.update_variance()
        if self.logging:
            self.log["W"].append(self.W)
            self.log["T"].append(self.T)
            self.log["sigma2"].append(self.sigma2)
            self.log["iteration"].append(self.iteration)

    def update_variance(self):
        """
        Update the variance of the mixture model using the new estimate of the deformable transformation.
        See the update rule for sigma2 in Eq. 23 of of https://arxiv.org/pdf/0905.2635.pdf.
        """

        # update objective function
        Lold = self.L
        self.L = (
            np.sum(np.log(self.Pden))
            + self.D * self.M * np.log(self.sigma2) / 2
            - self.lambdaFactor / 2 * np.trace(np.transpose(self.W) @ self.G @ self.W)
            - self.tauFactor / 2 * np.trace(np.transpose(self.T) @ self.Phi @ self.T)
        )

        self.diff = np.abs((self.L - Lold) / self.L)

        # update sigma
        yPy = np.dot(
            np.transpose(self.Pt1), np.sum(np.multiply(self.Y, self.Y), axis=1)
        )
        xPx = np.dot(np.transpose(self.P1), np.sum(np.multiply(self.T, self.T), axis=1))
        trPXY = np.sum(np.multiply(self.T, self.PY))
        self.sigma2 = (xPx - 2 * trPXY + yPy) / (self.Np * self.D)

        if self.sigma2 <= 0:
            self.sigma2 = self.tolerance / 10

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

    def getResults(self):
        result = {
            "X": self.X,
            "T": self.T,
            "W": self.W,
            "G": self.G,
            "sigma2": self.sigma2,
            "runtimes": self.runTimes,
        }
        if self.logging:
            result["log"] = self.log
        return result
