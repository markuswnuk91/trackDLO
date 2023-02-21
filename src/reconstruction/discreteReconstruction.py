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
    """
    Implementation of the a shape reconstruction based on the finite segment DLO model

    Attributes
        ----------

        wPosDiff: float
            factor for weighting the positional error term in the cost function

        correspondanceWeightingFactor:
            factor for weightin the correspondances between points X on the discrete representation and points Y on the continuous representation
    """

    def __init__(
        self,
        wPosDiff=None,
        correspondanceWeightingFactor=None,
        annealingFlex=None,
        annealingTor=None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        if (
            correspondanceWeightingFactor is not None
            and correspondanceWeightingFactor.size > self.Y.shape[0]
        ):
            raise ValueError(
                "Expected a diffrent number of weights ({}) than number of correspondances ({}). Expected same number for weights and correspondances.".format(
                    correspondanceWeightingFactor.size, self.Y.shape[0]
                )
            )
        self.wPosDiff = 1 if wPosDiff is None else wPosDiff
        self.correspondanceWeightingFactor = (
            np.ones(self.Y.shape[0])
            if correspondanceWeightingFactor is None
            else correspondanceWeightingFactor
        )
        self.annealingFlex = 1 if annealingFlex is None else annealingFlex
        self.annealingTor = 1 if annealingTor is None else annealingTor
        self.iter = 0
        self.optimVars, self.mappingDict = self.initOptimVars(lockedDofs=[3, 4, 5])

    #     self.optimVars, self.mappingDict = self.initOptimVars(
    #         **{
    #             # "x0": self.x0,
    #             "rot0": self.rot0,
    #             "alphas": self.alphas,
    #             "betas": self.betas,
    #             "gammas": self.gammas,
    #         }
    #     )

    # def initOptimVars(self, **kwargs):
    #     numOptVars = 0
    #     for key in kwargs:
    #         numOptVars += len(kwargs[key])
    #     optimVars = np.zeros(numOptVars)
    #     startOptVarIdx = 0
    #     endOptVarIdx = 0
    #     mappingDict = {}
    #     for key in kwargs:
    #         endOptVarIdx += len(kwargs[key])
    #         optimVars[startOptVarIdx:endOptVarIdx] = kwargs[key]
    #         mappingDict[key] = list(range(startOptVarIdx, endOptVarIdx))
    #         startOptVarIdx = endOptVarIdx
    #     return optimVars, mappingDict

    def initOptimVars(self, lockedDofs=None):
        """initializes the optimization variables

        Args:
            lockedDofs (list, optional): list of degrees of freedom to be locked during optimiation. Defaults to None.
        """
        mappingDict = {}
        if lockedDofs is None:
            optimVars = np.zeros(self.skel.getNumDofs())
            mappingDict["freeDofs"] = range(0, self.skel.getNumDofs())
        else:
            optimVars = np.zeros(self.skel.getNumDofs() - len(lockedDofs))
            mappingDict["lockedDofs"] = lockedDofs
            mappingDict["freeDofs"] = [
                index
                for index in range(0, self.skel.getNumDofs())
                if index not in lockedDofs
            ]
        return optimVars, mappingDict

    def updateParameters(self, optimVars):
        # # determine dofs from optimVars
        # if "x0" in self.mappingDict:
        #     self.x0 = optimVars[self.mappingDict["x0"]]
        # if "rot0" in self.mappingDict:
        #     self.rot0 = optimVars[self.mappingDict["rot0"]]
        # if "alphas" in self.mappingDict:
        #     self.alphas = optimVars[self.mappingDict["alphas"]]
        # if "betas" in self.mappingDict:
        #     self.betas = optimVars[self.mappingDict["betas"]]
        # if "gammas" in self.mappingDict:
        #     self.gammas = optimVars[self.mappingDict["gammas"]]
        # q = self.mapAnglesToDartPositions(
        #     self.x0, self.rot0, self.alphas, self.betas, self.gammas
        # )
        self.q[self.mappingDict["freeDofs"]] = optimVars

        (
            self.x0,
            self.rot0,
            self.alphas,
            self.betas,
            self.gammas,
        ) = self.mapDartPositionsToAngles(self.q)
        # update skeleton
        self.setPositions(self.q)

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
            self.wPosDiff
            * np.sum(
                self.correspondanceWeightingFactor
                * np.square(np.linalg.norm(self.Y - self.X, axis=1))
            )
            # + self.wPosDiff * (1 - np.exp(-np.linalg.norm(self.Y - self.X) / 1000))
            # + np.sum(self.aPsi)
        )
        return error

    def costFunJac(self, optimVars):
        # determine dofs from optimVars
        self.updateParameters(optimVars)
        # update skeleton
        self.setPositions(self.q)

        # determineJacobians
        jacobians = self.getJacobianFromLocalCoordinates(self.SY)

        # map jacobian entries to optimVars
        J = np.zeros(len(optimVars))
        for i, optimVar in enumerate(optimVars):
            correspondingDartIndex = self.mappingDict["freeDofs"][i]
            jacobianRows = []
            for jacobian in jacobians:
                # fill dart jacobians with zeros
                if jacobian.shape[1] < len(self.q):
                    paddedJacobian = np.pad(
                        jacobian,
                        ((0, 0), (0, len(self.q) - jacobian.shape[1] % len(self.q))),
                        "constant",
                    )
                elif jacobian.shape[1] == len(self.q):
                    paddedJacobian = jacobian
                else:
                    raise ValueError("Jacobina seems to have wrong dimension.")
                jacobianRows.append(paddedJacobian[3:6, correspondingDartIndex])
            jacobianMultiplikatorMatrix = np.vstack(jacobianRows)
            # calcualte cost function derivative
            J[i] = np.sum(
                -2
                * self.wPosDiff
                * self.correspondanceWeightingFactor
                * np.sum((self.Y - self.X) * jacobianMultiplikatorMatrix, axis=1)
            )

        return J

    def reconstructShape(self, numIter: int = None):
        res = least_squares(
            self.costFun,
            self.optimVars,
            self.costFunJac,
            max_nfev=numIter,
            verbose=2,
        )
