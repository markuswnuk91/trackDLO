from builtins import super
import os
import sys
import numpy as np
import numbers
from warnings import warn
from scipy.optimize import least_squares
import json

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

        weigthUFlex: float
            factor for weighting the flexibility error term in the cost function

        weigthUTor: float
            factor for weighting the torsional error term in the cost function

        weigthUGrav: float
            factor for weighting the gravitational error term in the cost function

        wPosDiff: float
            factor for weighting the positional error term in the cost function

        L: float
            Length of the DLO which should be reconstructed

        Sintegral: List
            List of local coordinate vectors the integral over zeta should be solved for.
            For each point in Y the integral needs to be solved.
    """

    def __init__(
        self,
        aPhi=None,
        aTheta=None,
        aPsi=None,
        Rflex=None,
        Rtor=None,
        Roh=None,
        wPosDiff=None,
        annealingFlex=None,
        annealingTor=None,
        *args,
        **kwargs
    ):
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

        self.aPhi = 0 * np.ones(self.N) if aPhi is None else aPhi
        self.aTheta = 0 * np.ones(self.N) if aTheta is None else aTheta
        self.aPsi = 0 * np.ones(self.N) if aPsi is None else aPsi
        self.Rflex = 1 if Rflex is None else Rflex
        self.Rtor = 1 if Rtor is None else Rtor
        self.Roh = 0.1 if Roh is None else Roh  # kg/m
        self.x0 = self.Y[0]
        self.wPosDiff = 1 if wPosDiff is None else wPosDiff
        self.annealingFlex = 1 if annealingFlex is None else annealingFlex
        self.annealingTor = 1 if annealingTor is None else annealingTor
        # self.lambdaPhi = 0.0001
        # self.lambdaTheta = 0.0001
        # self.lambdaPsi = 0.0001

        self.Ec = self.evalAnsatzFuns(self.Sc)
        self.Ex = self.evalAnsatzFuns(self.Sx)
        self.Sintegral = self.determineIntegrationPoints(self.Sc, self.Sx)
        self.i = 0
        self.optimVars, self.mappingDict = self.initOptimVars(
            **{"aPhi": self.aPhi, "aTheta": self.aTheta, "aPsi": self.aPsi}
        )
        self.paramDict = {}
        # self.optimVars, self.mappingDict = self.initOptimVars(
        #     **{"aPhi": self.aPhi, "aTheta": self.aTheta}
        # )
        self.beta = 0.1

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

    def evalThetaDervi_S(self, S):
        return self.aTheta @ self.evalAnsatzFunDerivs(S)

    def evalPhi(self, S):
        phi = self.aPhi @ self.evalAnsatzFuns(S)
        # print("Phi: {}".format(phi))
        return phi

    def evalPhiDeriv_S(self, S):
        return self.aPhi @ self.evalAnsatzFunDerivs(S)

    def evalPsi(self, S):
        psi = self.aPsi @ self.evalAnsatzFuns(S)
        # print("Phi: {}".format(phi))
        return psi

    def evalPsiDeriv_S(self, S):
        return self.aPsi @ self.evalAnsatzFunDerivs(S)

    def evalZeta(self, S):
        """from eq.5, where we assume zero strain"""
        return np.array(
            (
                np.sin(self.evalTheta(S)) * np.cos(self.evalPhi(S)),
                np.sin(self.evalTheta(S)) * np.sin(self.evalPhi(S)),
                np.cos(self.evalTheta(S)),
            )
        )

    # def evalZetaDeriv_S(self, S):
    #     return np.array(
    #         (
    #             np.cos(self.evalTheta(S))
    #             * np.cos(self.evalPhi(S))
    #             * (self.aTheta @ self.evalAnsatzFunDerivs(S))
    #             - np.sin(self.evalTheta(S))
    #             * np.sin(self.evalPhi(S))
    #             * (self.aPhi @ self.evalAnsatzFunDerivs(S)),
    #             np.cos(self.evalTheta(S))
    #             * np.sin(self.evalPhi(S))
    #             * (self.aTheta @ self.evalAnsatzFunDerivs(S))
    #             + np.sin(self.evalTheta(S))
    #             * np.cos(self.evalPhi(S))
    #             * (self.aPhi @ self.evalAnsatzFunDerivs(S)),
    #             -np.sin(self.evalTheta(S))
    #             * (self.aTheta @ self.evalAnsatzFunDerivs(S)),
    #         )
    #     )

    def evalZetaDeriv_aTheta(self, S):
        res = np.zeros([3, S.size])
        res[0, :] = np.cos(self.evalTheta(S)) * np.cos(self.evalPhi(S))
        res[1, :] = np.cos(self.evalTheta(S)) * np.sin(self.evalPhi(S))
        res[2, :] = -np.sin(self.evalTheta(S))
        return res

    def evalZetaDeriv_aPhi(self, S):
        res = np.zeros([3, S.size])
        res[0, :] = -np.sin(self.evalTheta(S)) * np.sin(self.evalPhi(S))
        res[1, :] = np.sin(self.evalTheta(S)) * np.cos(self.evalPhi(S))
        res[2, :] = 0
        return res

    def evalKappaSquared(self, S):
        dThetaSquared = np.square(self.aTheta @ self.evalAnsatzFunDerivs(S))
        dPhiSquared = np.square(self.aPhi @ self.evalAnsatzFunDerivs(S))
        return dThetaSquared + dPhiSquared * np.square(np.sin(self.evalTheta(S)))

    def evalKappaSquaredDeriv_aTheta(self, S):
        return 2 * (
            self.aTheta * np.square(self.evalAnsatzFunDerivs(S)).T
        ).T + np.square(self.aPhi @ self.evalAnsatzFunDerivs(S)) * 2 * np.sin(
            self.evalTheta(S)
        ) * np.cos(
            self.evalTheta(S)
        ) * self.evalAnsatzFuns(
            S
        )

    def evalKappaSquaredDeriv_aPhi(self, S):
        return (
            2
            * self.aPhi
            * (
                np.square(self.evalAnsatzFunDerivs(S))
                * np.square(np.sin(self.evalTheta(S)))
            ).T
        ).T

    def evalKappaSquaredDerivs(self, S):
        kappaSqaredJac = np.zeros([self.optimVars.size, S.size])
        if "aPhi" in self.mappingDict:
            kappaSqaredJac[
                self.mappingDict["aPhi"], :
            ] = self.evalKappaSquaredDeriv_aPhi(S)
        if "aTheta" in self.mappingDict:
            kappaSqaredJac[
                self.mappingDict["aTheta"], :
            ] = self.evalKappaSquaredDeriv_aTheta(S)
        return kappaSqaredJac

    def evalOmegaSquared(self, S):
        dPhi = self.aPhi @ self.evalAnsatzFunDerivs(S)
        dPsi = self.aPsi @ self.evalAnsatzFunDerivs(S)
        omegaSquared = np.square(dPhi * np.cos(self.evalTheta(S)) + dPsi)
        # print(omegaSquared)
        return omegaSquared

    def evalOmegaSquaredDeriv_aTheta(self, S):
        return (
            (
                2 * (self.evalPhiDeriv_S(S)) * np.cos(self.evalTheta(S))
                + self.evalPsiDeriv_S(S)
            )
            * self.evalPhiDeriv_S(S)
            * (-np.sin(self.evalTheta(S)))
            * self.evalAnsatzFuns(S)
        )

    def evalOmegaSquaredDeriv_aPhi(self, S):
        return (
            2
            * (
                self.evalPhiDeriv_S(S) * np.cos(self.evalTheta(S))
                + self.evalPsiDeriv_S(S)
            )
            * self.evalAnsatzFunDerivs(S)
            * np.cos(self.evalTheta(S))
        )

    def evalOmegaSquaredDeriv_aPsi(self, S):
        return (
            2
            * (
                self.evalPhiDeriv_S(S) * np.cos(self.evalTheta(S))
                + self.evalPsiDeriv_S(S)
            )
            * self.evalAnsatzFunDerivs(S)
        )

    def evalOmegaSquaredDerivs(self, S):
        omegaSqaredJac = np.zeros([self.optimVars.size, S.size])
        if "aPhi" in self.mappingDict:
            omegaSqaredJac[
                self.mappingDict["aPhi"], :
            ] = self.evalOmegaSquaredDeriv_aPhi(S)
        if "aTheta" in self.mappingDict:
            omegaSqaredJac[
                self.mappingDict["aTheta"], :
            ] = self.evalOmegaSquaredDeriv_aTheta(S)
        if "aPsi" in self.mappingDict:
            omegaSqaredJac[
                self.mappingDict["aPsi"], :
            ] = self.evalOmegaSquaredDeriv_aPsi(S)
        return omegaSqaredJac

    def evalPositions(self, S):
        Xintegral = self.integrateOverS(self.evalZeta, S).transpose()
        x0 = self.x0
        X = self.x0 + Xintegral
        return X.transpose()

    def evalPositionsDeriv_aTheta_i(self, S, i):
        dZeta_daTheta_i = (
            lambda s: self.evalZetaDeriv_aTheta(s) * self.evalAnsatzFuns(s)[i % self.N]
        )
        return self.integrateOverS(dZeta_daTheta_i, S)

    def evalPositionsDeriv_aPhi_i(self, S, i):
        dZeta_daPhi_i = (
            lambda s: self.evalZetaDeriv_aPhi(s) * self.evalAnsatzFuns(s)[i % self.N]
        )
        return self.integrateOverS(dZeta_daPhi_i, S)

    # def evalPositionsJac_aTheta(self, S):
    #     PosJac_aTheta = np.zeros([self.aTheta.size, S.size])
    #     for i in self.mappingDict["aTheta"]:
    #         dZeta_daTheta_i = (
    #             lambda s: np.sum(self.evalZetaDeriv_aTheta(s), axis=0)
    #             * self.evalAnsatzFuns(s)[i % self.N]
    #         )
    #         PosJac_aTheta[i % self.N, :] = self.integrateOverS(dZeta_daTheta_i, S)
    #     return PosJac_aTheta

    # def evalPositionsJac_aPhi(self, S):
    #     PosJac_aPhi = np.zeros([self.aPhi.size, S.size])
    #     for i in self.mappingDict["aPhi"]:
    #         dZeta_daPhi_i = (
    #             lambda s: np.sum(self.evalZetaDeriv_aPhi(s), axis=0)
    #             * self.evalAnsatzFuns(s)[i % self.N]
    #         )
    #         PosJac_aPhi[i % self.N, :] = self.integrateOverS(dZeta_daPhi_i, S)
    #     return PosJac_aPhi

    def evalPositionsJac(self, S):
        positionDiffJac = np.zeros(self.optimVars.size)
        if "aPhi" in self.mappingDict:
            for i in self.mappingDict["aPhi"]:
                positionDiffJac[i] = -2 * np.sum(
                    (self.Y - self.X)
                    * self.evalPositionsDeriv_aPhi_i(S, i % self.N).transpose()
                )
        if "aTheta" in self.mappingDict:
            for i in self.mappingDict["aTheta"]:
                positionDiffJac[i] = -2 * np.sum(
                    (self.Y - self.X)
                    * self.evalPositionsDeriv_aTheta_i(S, i % self.N).transpose()
                )
        # if "aPhi" in self.mappingDict:
        #     positionDiffJac[self.mappingDict["aPhi"]] = (
        #         -2 * np.sum(self.Y - self.X) * self.evalPositionsDeriv_aTheta_i(S, 0)
        #     )
        # if "aTheta" in self.mappingDict:
        #     positionDiffJac[self.mappingDict["aTheta"]] = (
        #         -2
        #         * np.sum(self.Y - self.X)
        #         * np.sum(self.evalPositionsJac_aTheta(S), axis=1)
        #     )

        return positionDiffJac

    def determineIntegrationPoints(self, Sc, Sx):
        Sintegral = []
        for sx in Sx:
            Sintegral.append(np.append(Sc[Sc < sx], sx))
        return Sintegral

    def evalUflex(self, s):
        Uflex = 0.5 * self.Rflex * self.integrateOverS(self.evalKappaSquared, s)
        # print(Uflex)
        return Uflex

    def evalUtor(self, s):
        Utor = 0.5 * self.Rtor * self.integrateOverS(self.evalOmegaSquared, s)
        return Utor

    def evalUgrav(self, s):
        Ugrav = self.Roh * self.integrateOverS(self.evalPositions, s)[2]
        return Ugrav

    def integrateOverS(self, fun, SLim):
        """returns the integrals over [0,sLim] of a function fun, the function is evaluated at all collocation points Sc up to the values sLim given as a list in SLim.

        Args:
            fun (function(S)): vector valued function that can be evaluated at different local coordinates, where the dimension of the result is ordered along the first dimension and the results for different S are ordered along the second dimension.
            SLim (np.array): array of local coodinates determining the upper limit of the integrals

        Returns:
            integrals( np.array): Dxdim(SLim) array of integals
        """
        # XS = np.array(len(Sx), self.D)
        # for i, s in enumerate(S):
        #     Ec = self.Ec[self.Sc < s]
        #     Ex = Ex(i)
        #     Eall = np.hstack(Ec, Ex)
        #     XS[i, 1] = np.sin(aTheta @ Eall) * np.cos(aPhi @ Eall)
        SIntervals = self.determineIntegrationPoints(self.Sc, SLim)
        results = []
        for Sintegral in SIntervals:
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

        if "aPhi" in self.mappingDict:
            self.aPhi = optimVars[self.mappingDict["aPhi"]]
        if "aTheta" in self.mappingDict:
            self.aTheta = optimVars[self.mappingDict["aTheta"]]
        if "aPsi" in self.mappingDict:
            self.aPsi = optimVars[self.mappingDict["aPsi"]]
        self.X = self.evalPositions(self.Sx).transpose()

    def costFun(self, optimVars):
        self.updateParameters(optimVars)

        error = (
            self.annealingFlex**self.i * self.evalUflex([self.L])
            + self.annealingTor**self.i * self.evalUtor([self.L])
            + self.evalUgrav([self.L])
            + self.wPosDiff * np.square(np.linalg.norm(self.Y - self.X))
            # + self.wPosDiff * (1 - np.exp(-np.linalg.norm(self.Y - self.X) / 1000))
            # + np.sum(self.aPsi)
        )

        if callable(self.callback):
            kwargs = {"X": self.X, "Y": self.Y, "fileName": "img_" + str(self.i)}
            self.i += 1
            self.callback(**kwargs)
            # print(self.estimateLength())

        # print("Theta: {}".format(self.evalTheta(self.Sx)))
        # print("Kappa: {}".format(self.evalKappaSquared(self.Sx)))
        # print("Uflex: {}".format(self.evalUflex([self.L])))
        # print("Utor: {}".format(self.evalUtor([self.L])))
        # print("Ugrav: {}".format(self.evalUgrav([self.L])))
        return error

    def costFunJac(self, optimVars):
        self.updateParameters(optimVars)
        jacobianUFelx = (
            self.annealingFlex**self.i
            * self.Rflex
            * np.trapz(self.evalKappaSquaredDerivs(self.Sc), self.Sc)
        )
        jacobianUTor = (
            self.annealingTor**self.i
            * self.Rtor
            * np.trapz(self.evalOmegaSquaredDerivs(self.Sc), self.Sc)
        )
        jacPosDiff = self.wPosDiff * self.evalPositionsJac(self.Sx)
        jacobian = jacobianUFelx + jacobianUTor + jacPosDiff
        return jacobian

    def initOptimVars(self, **kwargs):
        # optimVars = np.zeros(2 * self.N + 3)
        # optimVars[:3] = self.x0
        # optimVars[3 : self.N + 3] = self.aPhi
        # optimVars[self.N + 3 : 2 * self.N + 3] = self.aTheta
        numOptVars = 0
        for key in kwargs:
            numOptVars += len(kwargs[key])
        optimVars = np.zeros(numOptVars)

        startOptVarIdx = 0
        endOptVarIdx = 0
        mappingDict = {}
        for key in kwargs:
            endOptVarIdx += len(kwargs[key])
            optimVars[startOptVarIdx:endOptVarIdx] = kwargs[key]
            mappingDict[key] = list(range(startOptVarIdx, endOptVarIdx))
            startOptVarIdx = endOptVarIdx
        return optimVars, mappingDict

    def estimateShape(self, numIter=None):
        res = least_squares(
            self.costFun,
            self.optimVars,
            self.costFunJac,
            max_nfev=numIter,
            verbose=2,
        )
        self.W = res.x

    def estimateLength(self):
        return np.trapz(
            np.linalg.norm(self.evalZeta(self.Sintegral[-1]), axis=0),
            self.Sintegral[-1],
        )

    def registerCallback(self, callback):
        self.callback = callback

    def writeParametersToJson(
        self, savePath="src/plot/plotdata/", fileName="continuousDLO"
    ):

        if savePath is not None and type(savePath) is not str:
            raise ValueError(
                "Error saving parameter dict. The given path should be a string."
            )

        if fileName is not None and type(fileName) is not str:
            raise ValueError(
                "Error saving parameter dict. The given file name should be a string."
            )

        self.paramDict["aPhi"] = self.aPhi.tolist()
        self.paramDict["aTheta"] = self.aTheta.tolist()
        self.paramDict["aPsi"] = self.aPsi.tolist()
        self.paramDict["L"] = self.L
        self.paramDict["Rtor"] = self.Rtor
        self.paramDict["Rflex"] = self.Rflex
        self.paramDict["Roh"] = self.Roh
        self.paramDict["numAnsatzFuns"] = self.N

        with open(savePath + fileName + ".json", "w") as fp:
            json.dump(self.paramDict, fp)
