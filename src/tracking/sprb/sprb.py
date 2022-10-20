from builtins import super
import os
import sys
import numpy as np
import numbers

try:
    sys.path.append(os.getcwd().replace("/src/tracking/spr", ""))
    from src.tracking.spr.spr import StructurePreservedRegistration
except:
    print("Imports for SPR failed.")
    raise


class StructurePreservedRegistration4BDLO(StructurePreservedRegistration):
    """
    Implementation of the Structure Preserved Registration for Branched Deformable Linear Objects (SPRB) based on the the paper:
    M. Wnuk, C. Hinze, M. Zürn, Q. Pan, A. Lechler and A. Verl, "Tracking Branched Deformable Linear Objects With Structure Preserved Registration by Branch-wise Probability Modification," 2021 27th International Conference on Mechatronics and Machine Vision in Practice (M2VIP), 2021, pp. 101-108
    Based on their provided implementation:
    https://github.com/markuswnuk91/M2VIP21-SPR4BranchedDLO

    Attributes
    ----------
    B: List of Lists
        B is the set of branches of a BDLO. For every branch of the BDLO B contains a list with the indices of the points in the source point set X that belong to this branch. So if a BDLO consists of three branches: len(B) = 3.
    """

    def __init__(self, B=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if B is not None and (not isinstance(B, list)):
            raise ValueError(
                "Expected a list of lists for branch set B. Instead got: {}".format(B)
            )

        self.B = [[x] for x in range(0, self.N)] if B is None else B
        self.K = len(self.B)
        self.PK = 1 / self.K * np.ones((self.N, self.M))
        self.PB = 1 / self.K * np.ones((self.K, 1))
        self.PNk = np.zeros((self.N, self.M))
        self.Nk = []
        for idxList in self.B:
            Nk = len(idxList)
            self.Nk.append(Nk)
            self.PNk[idxList, :] = 1 / Nk

    def estimateCorrespondance(self):
        """
        E-step: Compute the expectation step  of the EM algorithm.
        """
        P = np.sum((self.Y[None, :, :] - self.T[:, None, :]) ** 2, axis=2)

        c = (2 * np.pi * self.sigma2) ** (self.D / 2)
        c = c * self.mu / (1 - self.mu)
        c = c * self.N / self.M

        P = np.exp(-P / (2 * self.sigma2))
        Pmod = self.PK * self.PNk * P
        den = np.sum(Pmod, axis=0)
        den = np.tile(den, (self.N, 1))
        den[den == 0] = np.finfo(float).eps
        den += c

        self.Pden = den[0, :]
        self.P = np.divide(Pmod, den)
        self.Pt1 = np.sum(self.P, axis=0)
        self.P1 = np.sum(self.P, axis=1)
        self.Np = np.sum(self.P1)
        self.PY = np.matmul(self.P, self.Y)

    def updateParameters(self):
        """
        M-step: Calculate a new parameters of the registration.
        """

        tauFactor = (self.tauAnnealing) ** (self.iteration) * self.tauFactor
        lambdaFactor = (self.lambdaAnnealing) ** (self.iteration) * self.lambdaFactor

        dP1 = np.diag(self.P1)
        A = (
            np.dot(dP1, self.G)
            + lambdaFactor * self.sigma2 * np.eye(self.N)
            + tauFactor * self.sigma2 * np.dot(self.Phi, self.G)
        )
        B = (
            self.PY
            - np.dot(dP1, self.X)
            - tauFactor * self.sigma2 * np.dot(self.Phi, self.X)
        )
        self.W = np.linalg.solve(A, B)

        # set the new targets
        self.computeTargets()

        # update objective function
        Lold = self.L
        self.L = (
            np.sum(np.log(self.Pden))
            + self.D * self.M * np.log(self.sigma2) / 2
            - lambdaFactor / 2 * np.trace(np.transpose(self.W) @ self.G @ self.W)
            - tauFactor / 2 * np.trace(np.transpose(self.T) @ self.Phi @ self.T)
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

        # update branch probability
        for i, idxList in enumerate(self.B):
            self.PB[i] = 1 / self.N * np.sum(self.P[idxList, :])
            self.PK[idxList, :] = self.PB[i]
