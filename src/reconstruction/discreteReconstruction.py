from builtins import super
import os
import sys
import numpy as np
import numbers
from warnings import warn
from scipy.optimize import least_squares
import json

try:
    sys.path.append(os.getcwd().replace("/src/reconstruction", ""))
    from src.reconstruction.shapeReconstruction import ShapeReconstruction
    from src.modelling.finiteSegmentModel import FiniteSegmentModel
except:
    print("Imports for discrete shape reconstruction failed.")
    raise


class DiscreteReconstruction(ShapeReconstruction, FiniteSegmentModel):
    def __init__(
        self, wPosDiff=None, annealingFlex=None, annealingTor=None, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.wPosDiff = 1 if wPosDiff is None else wPosDiff
        self.annealingFlex = 1 if annealingFlex is None else annealingFlex
        self.annealingTor = 1 if annealingTor is None else annealingTor
        self.iter = 0
        self.optimVars, self.mappingDict = self.initOptimVars(
            **{
                # "x0": self.x0,
                "rot0": self.rot0,
                "alphas": self.alphas,
                "betas": self.betas,
                "gammas": self.gammas,
            }
        )

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

    def updateParameters(self, optimVars):
        # determine dofs from optimVars
        if "x0" in self.mappingDict:
            self.x0 = optimVars[self.mappingDict["x0"]]
        if "rot0" in self.mappingDict:
            self.rot0 = optimVars[self.mappingDict["rot0"]]
        if "alphas" in self.mappingDict:
            self.alphas = optimVars[self.mappingDict["alphas"]]
        if "betas" in self.mappingDict:
            self.betas = optimVars[self.mappingDict["betas"]]
        if "gammas" in self.mappingDict:
            self.gammas = optimVars[self.mappingDict["gammas"]]
        q = self.mapAnglesToDartPositions(
            self.x0, self.rot0, self.alphas, self.betas, self.gammas
        )

        # update skeleton
        self.setPositions(q)

        # determine Positions
        self.X = self.getCaresianPositionsFromLocalCoordinates(self.SY)

        # update Iteration Number
        self.iter += 1

        if callable(self.callback):
            self.callback()

    def costFun(self, optimVars):
        self.updateParameters(optimVars)
        error = (
            # self.annealingFlex**self.iter * self.evalUflex(self.L, self.numSc)
            # + self.annealingTor**self.iter * self.evalUtor(self.L, self.numSc)
            # + self.evalUgrav(self.L, self.numSc)
            +self.wPosDiff
            * np.square(np.linalg.norm((self.Y - self.X)))
            # * np.square(
            #     np.sum(
            #         np.linspace(1, 0.5, self.Y.shape[0])
            #         * np.linalg.norm((self.Y - self.X), axis=1)
            #     )
            # + self.wPosDiff * (1 - np.exp(-np.linalg.norm(self.Y - self.X) / 1000))
            # + np.sum(self.aPsi)
        )
        return error

    def reconstructShape(self, numIter: int = None):
        res = least_squares(
            self.costFun,
            self.optimVars,
            max_nfev=numIter,
            verbose=2,
        )
