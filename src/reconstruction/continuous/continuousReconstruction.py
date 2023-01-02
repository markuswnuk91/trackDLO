from builtins import super
import os
import sys
import numpy as np
import numbers
from warnings import warn
from scipy.optimize import least_squares
import json

try:
    sys.path.append(os.getcwd().replace("/src/reconstruction/coninuous", ""))
    from src.reconstruction.shapeReconstruction import ShapeReconstruction
    from src.modelling.wakamatsuModel import WakamatsuModel
except:
    print("Imports for coninuous shape reconstruction failed.")
    raise


class ContinuousReconstruction(ShapeReconstruction, WakamatsuModel):
    """
    Implementation of the a shape reconstruction based on the differential geometry DLO model by H. Wakamatsu and S. Hirai from the paper:
    Wakamatsu H, Hirai S. Static Modeling of Linear Object Deformation Based on Differential Geometry. The International Journal of Robotics Research. 2004;23(3):293-311
    Note: The paper by Wakamatsu et al uses local coordinates s in [0,L], while this class uses normalized local coordinates s in [0,1].

    Attributes
        ----------

        numSc: int
             Number of collocation points along the DLO used to sample the continuous shape for integration

        Sc: numpy array
            (numSc)x1 array of local coordinates of collocation points along the DLO used to sample the continous shape.

        SY: numpy array
            Mx1 array of normalized local coordinates in [0,1] corresponding to the Y target points.

        Sy: numpy array
            Mx1 array of local coordinates in [0,L] corresponding to the Y target points.

        weigthUFlex: float
            factor for weighting the flexibility error term in the cost function

        weigthUTor: float
            factor for weighting the torsional error term in the cost function

        weigthUGrav: float
            factor for weighting the gravitational error term in the cost function

        wPosDiff: float
            factor for weighting the positional error term in the cost function

        Sintegral: List
            List of local coordinate vectors the integral over zeta should be solved for.
            For each point in Y the integral needs to be solved.
    """

    def __init__(
        self,
        numSc=None,
        wPosDiff=None,
        annealingFlex=None,
        annealingTor=None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        if numSc is not None and (not isinstance(numSc, numbers.Number) or numSc < 2):
            raise ValueError(
                "Expected a positive integer of at least two sample poitns for S instead got: {}".format(
                    numSc
                )
            )
        elif isinstance(numSc, numbers.Number) and not isinstance(numSc, int):
            warn(
                "Received a non-integer value for number of collocations (S): {}. Casting to integer.".format(
                    numSc
                )
            )
            numSc = int(numSc)
        elif isinstance(numSc, numbers.Number) and numSc < self.SY.size:
            warn(
                "Received less collocation points than local coodinates corresponding to the target points: {}. Increasing the number of collocation points to match the number of tartget points.".format(
                    numSc
                )
            )
            numSc = int(self.Sy.size)

        self.numSc = 100 if numSc is None else numSc
        self.wPosDiff = 1 if wPosDiff is None else wPosDiff
        self.annealingFlex = 1 if annealingFlex is None else annealingFlex
        self.annealingTor = 1 if annealingTor is None else annealingTor
        self.iter = 0
        self.Sc = np.linspace(0, self.L, self.numSc)
        self.Sy = self.SY * self.L
        self.numIntegrationPoints = self.determineNumIntegrationPointsAsList(
            self.Sy, self.numSc
        )
        self.X = self.evalPositions(self.Sy, self.numIntegrationPoints)
        self.optimVars, self.mappingDict = self.initOptimVars(
            **{"aPhi": self.aPhi, "aTheta": self.aTheta, "aPsi": self.aPsi}
        )

    def determineIntegrationPoints(self, Sc, Sy):
        Sintegral = []
        for sx in Sy:
            Sintegral.append(np.append(Sc[Sc < sx], sx))
        return Sintegral

    def determineNumIntegrationPoints(self, s, numSc):
        """returns the number of integration points for integrationg the position x of a DLO corresponding to the local coordinate sx, given the number of collocation points numSc.

        Args:
            s (float): normalized local coordintate in [0,1]
            numSc (int): number of collocation points the DLO is discretized into for numerical integration.

        Returns:
            int: number of evaluation points for the integral
        """
        return int(numSc * s)

    def determineNumIntegrationPointsAsList(self, S, numSc):
        """returns a list with the number of integration points for integrationg the positions X of a DLO corresponding to the local coordinates in Sx, given the number of collocation points numSc.

        Args:
            S (array): normalized local coordintates in [0,1]
            numSc (int): number of collocation points the DLO is discretized into for numerical integration.

        Returns:
            int: number of evaluation points for the integral
        """
        numIntegrationPointList = []
        for s in S:
            numIntegrationPointList.append(self.determineNumIntegrationPoints(s, numSc))
        return numIntegrationPointList

    def initOptimVars(self, **kwargs):
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

    def integrateOverS(self, fun, SLim):
        """returns the integrals over [0,sLim] of a function fun, the function is evaluated at all collocation points Sc up to the values sLim given as a list in SLim.

        Args:
            fun (function(S)): vector valued function that can be evaluated at different local coordinates, where the dimension of the result is ordered along the first dimension and the results for different S are ordered along the second dimension.
            SLim (np.array): array of local coodinates determining the upper limit of the integrals

        Returns:
            integrals( np.array): Dxdim(SLim) array of integals
        """
        SIntervals = self.determineIntegrationPoints(self.Sc, SLim)
        results = []
        for Sintegral in SIntervals:
            funValues = fun(Sintegral)
            integral = np.trapz(funValues, Sintegral)
            results.append(integral)
        return np.array(results).transpose()

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
        return positionDiffJac

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

    def updateParameters(self, optimVars):
        if "aPhi" in self.mappingDict:
            self.aPhi = optimVars[self.mappingDict["aPhi"]]
        if "aTheta" in self.mappingDict:
            self.aTheta = optimVars[self.mappingDict["aTheta"]]
        if "aPsi" in self.mappingDict:
            self.aPsi = optimVars[self.mappingDict["aPsi"]]
        self.X = self.evalPositions(self.Sy, self.numIntegrationPoints)
        self.iter += 1

    def costFun(self, optimVars):
        self.updateParameters(optimVars)
        error = (
            self.annealingFlex**self.iter * self.evalUflex(self.L, self.numSc)
            + self.annealingTor**self.iter * self.evalUtor(self.L, self.numSc)
            + self.evalUgrav(self.L, self.numSc)
            + self.wPosDiff * np.square(np.linalg.norm(self.Y - self.X))
            # + self.wPosDiff * (1 - np.exp(-np.linalg.norm(self.Y - self.X) / 1000))
            # + np.sum(self.aPsi)
        )
        if callable(self.callback):
            self.callback()
        return error

    def costFunJac(self, optimVars):
        jacobianUFelx = (
            self.annealingFlex**self.iter
            * self.Rflex
            * np.trapz(self.evalKappaSquaredDerivs(self.Sc), self.Sc)
        )
        jacobianUTor = (
            self.annealingTor**self.iter
            * self.Rtor
            * np.trapz(self.evalOmegaSquaredDerivs(self.Sc), self.Sc)
        )
        jacPosDiff = self.wPosDiff * self.evalPositionsJac(self.Sy)
        jacobian = jacobianUFelx + jacobianUTor + jacPosDiff
        return jacobian

    def reconstructShape(self, numIter=None):
        res = least_squares(
            self.costFun,
            self.optimVars,
            self.costFunJac,
            max_nfev=numIter,
            verbose=2,
        )

    def estimateLength(self):
        return np.trapz(
            np.linalg.norm(self.evalZeta(self.Sintegral[-1]), axis=0),
            self.Sintegral[-1],
        )

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

        paramDict = {}
        paramDict["aPhi"] = self.aPhi.tolist()
        paramDict["aTheta"] = self.aTheta.tolist()
        paramDict["aPsi"] = self.aPsi.tolist()
        paramDict["L"] = self.L
        paramDict["Rtor"] = self.Rtor
        paramDict["Rflex"] = self.Rflex
        paramDict["Roh"] = self.Roh
        paramDict["numAnsatzFuns"] = self.N

        with open(savePath + fileName + ".json", "w") as fp:
            json.dump(paramDict, fp)
