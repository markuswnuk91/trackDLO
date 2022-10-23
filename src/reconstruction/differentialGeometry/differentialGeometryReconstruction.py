from builtins import super
import os
import sys
import numpy as np
import numbers
from warnings import warn
from scipy.optimize import least_squares

try:
    sys.path.append(os.getcwd().replace("/src/reconstruction/differentialGeometry", ""))
    from src.reconstruction.shapeReconstruction import ShapeReconstruction
except:
    print("Imports for differential geometry shape reconstruction failed.")
    raise


class DifferentialGeometryReconstruction(ShapeReconstruction):
    """
    Implementation of the a shape reconstruction based on the differential geometry DLO model by H. Wakamatsu and S. Hirai from the paper:
    Wakamatsu H, Hirai S. Static Modeling of Linear Object Deformation Based on Differential Geometry. The International Journal of Robotics Research. 2004;23(3):293-311

    Attributes
        ----------
        N: int
            Number of ansatz functions

        W: numpy array
            NxD array of weights. Provide as input to initialize weights. Otherwise defaults to all zero.

        L: float
            Length of the DLO which should be reconstructed

        Sintegral: List
            List of local coodiante vectors the integral over zeta should be solved for.
            For each point in X the integral needs to be solved.
    """

    def __init__(self, L, N=None, W=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if N is not None and (not isinstance(N, numbers.Number) or N < 4 or N % 2 != 0):
            raise ValueError(
                "Expected a positive even integer larger than 4 for the number of ansazt functions. Using less ansatz functions is not useful for the Wakamatsu model. Instead got: {}".format(
                    N
                )
            )
        elif isinstance(N, numbers.Number) and not isinstance(N, int):
            warn(
                "Received a non-integer value for max_iterations: {}. Casting to integer.".format(
                    N
                )
            )
            N = int(N)

        # if W is not None and W.shape[0] != 2:
        #     raise ValueError(
        #         "First dimension of weight array must have the 2 entries. One for every angle (phi, theta) to be estimated."
        #     )

        # if W is not None and W.shape[1] != N:
        #     raise ValueError(
        #         "Second dimension of weight array must have the same number of entries as the number of ansatz functions."
        #     )

        if L is not None and (not isinstance(L, numbers.Number) or L < 0):
            raise ValueError(
                "Expected a positive float for length of the DLO. Instead got: {}".format(
                    L
                )
            )

        self.N = 10 if N is None else N
        self.W = np.zeros(3 + 2 * self.N) if W is None else W
        self.L = L
        self.Ec = self.evalAnsatzFuns(self.Sc)
        self.Ex = self.evalAnsatzFuns(self.Sx)
        self.Zeta = np.array((1, 0, 1))
        self.aPhi = self.W[3 : self.N + 3]
        self.aTheta = self.W[self.N + 3 : 2 * self.N + 3]
        # self.x0 = np.zeros(3)
        self.x0 = self.X[0]
        self.Sintegral = self.determineIntegrationPoints(self.Sc, self.Sx)
        self.optimVars = self.initOptimVars()

    def evalAnsatzFuns(self, S):
        """returns the ansatz functions evaluated at the local coodinates in S

        Args:
            S (np.array): Array of local coordinates in [0,1] where the ansatz functions should be evaluated

        Returns:
            E (np.array): NxD array of ansatz functions evaluated at local coodinates in S
        """
        E = np.ones((self.N, len(S)))
        E[1, :] = S * self.L
        for i in range(1, int((self.N / 2))):
            E[2 * i, :] = np.sin(2 * np.pi * i * S)
            E[2 * i + 1, :] = np.cos(2 * np.pi * i * S)
        return E

    def evalTheta(self, S):
        return self.aTheta @ self.evalAnsatzFuns(S)

    def evalPhi(self, S):
        return self.aPhi @ self.evalAnsatzFuns(S)

    def evalZeta(self, S):
        """from eq.5, where we assume zero strain"""
        return np.array(
            (
                np.sin(self.evalTheta(S)) * np.cos(self.evalPhi(S)),
                np.sin(self.evalTheta(S)) * np.sin(self.evalPhi(S)),
                np.cos(self.evalTheta(S)),
            )
        )

    def determineIntegrationPoints(self, Sc, Sx):
        Sintegral = []
        for sx in Sx:
            Sintegral.append(np.append(Sc[Sc < sx], sx))
        return Sintegral

    def integrateZeta(self, Sintegal):
        # XS = np.array(len(Sx), self.D)
        # for i, s in enumerate(S):
        #     Ec = self.Ec[self.Sc < s]
        #     Ex = Ex(i)
        #     Eall = np.hstack(Ec, Ex)
        #     XS[i, 1] = np.sin(aTheta @ Eall) * np.cos(aPhi @ Eall)
        zetaIntegral = np.zeros((len(Sintegal), self.D))
        for i, S in enumerate(Sintegal):
            for d in range(self.D):
                zetaIntegral[i, d] = np.trapz(self.evalZeta(S)[d, :], S)
        return zetaIntegral

    def updateParameters(self, optimVars):
        self.x0 = optimVars[:3]
        self.aPhi = optimVars[3 : self.N + 3]
        self.aTheta = optimVars[self.N + 3 : 2 * self.N + 3]

    def costFun(self, optimVars):
        self.updateParameters(optimVars)
        error = np.linalg.norm(
            self.X
            - (
                np.tile(optimVars[:3], (len(self.X), 1))
                + self.L * self.integrateZeta(self.Sintegral)
            )
        )
        print(error)
        return error

    def initOptimVars(self):
        optimVars = np.zeros(2 * self.N + 3)
        optimVars[:3] = self.x0
        optimVars[3 : self.N + 3] = self.aPhi
        optimVars[self.N + 3 : 2 * self.N + 3] = self.aTheta
        return optimVars

    def estimateShape(self):
        res = least_squares(self.costFun, self.optimVars, verbose=2)
        self.W = res.x
