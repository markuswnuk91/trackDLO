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
            Number of ansatz functions. Wakamatsu et al. use N = 10 in their implementation.

        aPhi: numpy array
            Nx1 array of initial values for the weights to approximate angle Phi. Provide as input to initialize weights. Otherwise defaults to all zero.

        L: float
            Length of the DLO which should be reconstructed

        Sintegral: List
            List of local coodiante vectors the integral over zeta should be solved for.
            For each point in Y the integral needs to be solved.
    """

    def __init__(self, aPhi=None, aTheta=None, Rflex=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # if W is not None and W.shape[0] != 2:
        #     raise ValueError(
        #         "First dimension of weight array must have the 2 entries. One for every angle (phi, theta) to be estimated."
        #     )

        # if W is not None and W.shape[1] != N:
        #     raise ValueError(
        #         "Second dimension of weight array must have the same number of entries as the number of ansatz functions."
        #     )

        self.N = 10

        self.aPhi = 0.001 * np.ones(self.N) if aPhi is None else aPhi
        self.aTheta = 0.001 * np.ones(self.N) if aTheta is None else aTheta
        self.x0 = self.Y[0]
        self.Rflex = 0.1 if Rflex is None else Rflex

        self.Ec = self.evalAnsatzFuns(self.Sc)
        self.Ex = self.evalAnsatzFuns(self.Sx)
        self.Sintegral = self.determineIntegrationPoints(self.Sc, self.Sx)
        self.optimVars = self.initOptimVars()

    def evalAnsatzFuns(self, S):
        """returns the ansatz functions evaluated at the local coodinates in S

        Args:
            S (np.array): Array of local coordinates in [0,L] where the ansatz functions should be evaluated

        Returns:
            E (np.array): NxD array of ansatz functions evaluated at local coodinates in S
        """
        E = np.ones((self.N, len(S)))
        E[1, :] = S / self.L
        for i in range(1, int((self.N / 2))):
            E[2 * i, :] = np.sin(2 * np.pi * i * S / self.L)
            E[2 * i + 1, :] = np.cos(2 * np.pi * i * S / self.L)
        return E

    def evalAnsatzFunDerivs(self, S):
        """returns the derivatives of the ansatz functions evaluated at the local coodinates in S

        Args:
            S (np.array): Array of local coordinates in [0,L] where the ansatz functions should be evaluated

        Returns:
            dE (np.array): NxD array of derivatives of the ansatz functions evaluated at local coodinates in S
        """
        dE = np.zeros((self.N, len(S)))
        dE[1, :] = np.ones(len(S)) / self.L
        for i in range(1, int((self.N / 2))):
            dE[2 * i, :] = np.cos(2 * np.pi * i * S / self.L) * (2 * np.pi * i / self.L)
            dE[2 * i + 1, :] = -np.sin(2 * np.pi * i * S / self.L) * (
                2 * np.pi * i / self.L
            )
        return dE

    def evalTheta(self, S):
        theta = self.aTheta @ self.evalAnsatzFuns(S)
        # print("Theta: {}".format(theta))
        return theta

    def evalPhi(self, S):
        phi = self.aPhi @ self.evalAnsatzFuns(S)
        # print("Phi: {}".format(phi))
        return phi

    def evalZeta(self, S):
        """from eq.5, where we assume zero strain"""
        return np.array(
            (
                np.sin(self.evalTheta(S)) * np.cos(self.evalPhi(S)),
                np.sin(self.evalTheta(S)) * np.sin(self.evalPhi(S)),
                np.cos(self.evalTheta(S)),
            )
        )

    def evalKappa(self, S):
        dThetaSquared = np.square(self.aTheta @ self.evalAnsatzFunDerivs(S))
        dPhiSquared = np.square(self.aPhi @ self.evalAnsatzFunDerivs(S))
        return dThetaSquared + dPhiSquared * np.square(np.sin(self.evalTheta(S)))

    def determineIntegrationPoints(self, Sc, Sx):
        Sintegral = []
        for sx in Sx:
            Sintegral.append(np.append(Sc[Sc < sx], sx))
        return Sintegral

    def evalUflex(self, s):
        Uflex = self.Rflex * self.integrateOverS(self.evalKappa, s)
        # print(Uflex)
        return Uflex

    def integrateOverS(self, fun, SLim):
        """returns the integrals over [0,sLim] of a function fun, the function is evaluated at all collocation points Sc up to the values sLim given as a list in SLim.

        Args:
            fun (function(S)): funcntion that can be evaluated at different local coordinates, where teh results for different s are ordered along the second dimenstion.
            SLim (np.array): array of local coodinates determining the upper limit of the integrals

        Returns:
            integrals( np.array): 1xD array
        """
        # XS = np.array(len(Sx), self.D)
        # for i, s in enumerate(S):
        #     Ec = self.Ec[self.Sc < s]
        #     Ex = Ex(i)
        #     Eall = np.hstack(Ec, Ex)
        #     XS[i, 1] = np.sin(aTheta @ Eall) * np.cos(aPhi @ Eall)
        SIntervals = self.determineIntegrationPoints(self.Sc, SLim)
        results = []
        for i, Sintegral in enumerate(SIntervals):
            funValues = fun(Sintegral)
            integral = np.trapz(funValues, Sintegral)
            results.append(integral)
        return np.array(results).transpose()

    # def integrateZeta(self, Sintegral):
    #     zetaIntegral = np.zeros((len(Sintegral), self.D))
    #     for i, S in enumerate(Sintegral):
    #         for d in range(self.D):
    #             zetaIntegral[i, d] = np.trapz(self.evalZeta(S)[d, :], S)
    #     return zetaIntegral

    def updateParameters(self, optimVars):
        # self.x0 = optimVars[:3]
        # self.aPhi = optimVars[3 : self.N + 3]
        # self.aTheta = optimVars[self.N + 3 : 2 * self.N + 3]

        self.aPhi = optimVars[0 : self.N]
        self.aTheta = optimVars[self.N : 2 * self.N]
        self.X = self.x0 + self.integrateOverS(self.evalZeta, self.Sx).transpose()

    def costFun(self, optimVars):
        self.updateParameters(optimVars)
        error = self.evalUflex([self.L]) + np.linalg.norm(self.Y - self.X)

        if callable(self.callback):
            kwargs = {
                "X": self.X,
                "Y": self.Y,
            }
            self.callback(**kwargs)
        return error

    def initOptimVars(self):
        # optimVars = np.zeros(2 * self.N + 3)
        # optimVars[:3] = self.x0
        # optimVars[3 : self.N + 3] = self.aPhi
        # optimVars[self.N + 3 : 2 * self.N + 3] = self.aTheta

        optimVars = np.zeros(2 * self.N)
        optimVars[0 : self.N] = self.aPhi
        optimVars[self.N : 2 * self.N] = self.aTheta
        return optimVars

    def estimateShape(self):
        res = least_squares(self.costFun, self.optimVars, verbose=2)
        self.W = res.x

    def registerCallback(self, callback):
        self.callback = callback