import os
import sys
import numpy as np
import numbers
from warnings import warn
from scipy.optimize import least_squares

try:
    sys.path.append(os.getcwd().replace("/src/reconstruction", ""))
    from src.reconstruction.shapeReconstruction import ShapeReconstruction
    from src.simulation.bdlo import BranchedDeformableLinearObject
    from src.localization.correspondanceEstimation.topologyBasedCorrespondanceEstimation import (
        TopologyBasedCorrespondanceEstimation,
    )
except:
    print("Imports for discrete shape reconstruction failed.")
    raise


class BDLOLocalization(TopologyBasedCorrespondanceEstimation):
    """Class for perfoming a localization for a BDLO from point cloud data
    Localization aims to obtain the generalized coordinates corresponding to the current configuration of the BDLO from a (high dimensional) set of points (point cloud).
    This class uses a discrete finite segment BDLO model.

    Attributes:
    -------------
    bdlo (BranchedDeformableLinearObject):
        Finite segment model of the branched deformable linear object which provides an interface to obatain the positions and jacobians corresponding to a local coordinate in each branch of the BDLO

    C (np.ndarray):
        2D corresponance matrix which orders the points in X such that they correspond to the points in Y according to: C@X ~ Y

    K (int):
        number of branches

    YTarget (np.array):
        sampled Y positions on the extracted topology which are the target positions for the optimization
    """

    def __init__(self, S, callback=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if np.any(S > 1) or np.any(S < 0):
            raise ValueError(
                "Obtained non-nomalized coordinates for local coordinates. Expected values to be normalized by the length of the DLO in [0,1]"
            )
        self.S = S
        self.callback = None if callback is None else callback

        self.iter = 0
        self.bdlo = self.templateTopology
        self.Ns = len(self.S)
        self.q = self.bdlo.skel.getPositions()
        self.K = self.bdlo.getNumBranches()
        self.X = np.zeros((self.Ns, self.D))
        (self.N, _) = self.X.shape

        # unlockedDofs = []
        # for i, branch in enumerate(self.bdlo.getBranches()):
        #     branchRootDofIndices = self.bdlo.getBranchRootDofIndices(i)
        #     for branchRootDofIndex in branchRootDofIndices:
        #         unlockedDofs.append(branchRootDofIndex)
        # lockedDofs = list(range(0, self.bdlo.skel.getNumDofs()))
        # for dof in lockedDofs:
        #     if dof in unlockedDofs:
        #         lockedDofs.remove(dof)
        # self.optimVars, self.mappingDict = self.initOptimVars(lockedDofs)
        self.optimVars, self.mappingDict = self.initOptimVars(None)

    def sampleCartesianPositionsFromModel(self):
        X = np.zeros((self.Ns * self.K, self.D))
        for b in range(0, self.K):
            X[
                b * self.Ns : (b + 1) * self.Ns
            ] = self.bdlo.getCartesianPositionsFromBranchLocalCoordinates(b, self.S)
        return X

    def initOptimVars(self, lockedDofs=None):
        """initializes the optimization variables

        Args:
            lockedDofs (list, optional): list of degrees of freedom to be locked during optimiation. Defaults to None.
        """
        mappingDict = {}
        if lockedDofs is None:
            optimVars = self.q
            mappingDict["freeDofs"] = range(0, self.bdlo.skel.getNumDofs())
        else:
            mappingDict["lockedDofs"] = lockedDofs
            mappingDict["freeDofs"] = [
                index
                for index in range(0, self.bdlo.skel.getNumDofs())
                if index not in lockedDofs
            ]
            optimVars = self.q[mappingDict["freeDofs"]]
        return optimVars, mappingDict

    def updateParameters(self, optimVars):
        # update skeleton
        self.q[self.mappingDict["freeDofs"]] = optimVars
        self.bdlo.skel.setPositions(self.q)

        # determine Positions
        self.X = self.sampleCartesianPositionsFromModel()

    def costFun(self, optimVars):
        self.updateParameters(optimVars)
        # for i, y in enumerate(self.Y):
        #     correspondingBranchIndex = self.CBY[i]
        #     correspondingLocalCoordinate = self.S[i]
        #     x = self.bdlo.getCartesianPositionFromBranchLocalCoordinate(
        #         correspondingBranchIndex, correspondingLocalCoordinate
        #     )
        #     errors[i] = np.linalg.norm(y - x)
        #     self.X[i] = x
        error = np.sum(
            np.square(np.linalg.norm(self.YTarget - self.C @ self.X, axis=1))
        )
        return error

    def costFunJac(self, optimVars):
        # determine dofs from optimVars
        self.updateParameters(optimVars)

        # determineJacobians
        jacobians = []
        for b in range(0, self.K):
            for s in self.S:
                jacobians.append(self.bdlo.getJacobianFromBranchLocalCoordinate(b, s))

        # map jacobian entries to optimVars
        J = np.zeros(len(optimVars))
        for i, optimVar in enumerate(optimVars):
            correspondingDartIndex = self.mappingDict["freeDofs"][i]
            jacobianRows = []
            for jacobian in jacobians:
                # # fill dart jacobians with zeros
                # if jacobian.shape[1] < len(self.q):
                #     paddedJacobian = np.pad(
                #         jacobian,
                #         ((0, 0), (0, len(self.q) - jacobian.shape[1] % len(self.q))),
                #         "constant",
                #     )
                # elif jacobian.shape[1] == len(self.q):
                #     paddedJacobian = jacobian.copy()
                # else:
                #     raise ValueError("Jacobian seems to have wrong dimension.")
                # jacobianRows.append(paddedJacobian[3:6, correspondingDartIndex])
                jacobianRows.append(jacobian[3:6, correspondingDartIndex])
            jacobianMultiplikatorMatrix = np.vstack(jacobianRows)
            # calcualte cost function derivative

            J[i] = np.sum(
                -2
                * np.sum(
                    (self.YTarget - self.C @ self.X)
                    * (self.C @ jacobianMultiplikatorMatrix),
                    axis=1,
                )
            )
        # update Iteration Number
        self.iter += 1

        if callable(self.callback):
            self.callback()
        return J

    def reconstructShape(self, numIter: int = -1, verbose=0):
        if numIter == -1:
            numIter = None
        if self.extractedTopology is None:
            warn("No topology yet extracted. Extracting topology ...")
            self.extractTopology()
        (self.YTarget, self.X, self.C) = self.findCorrespondancesFromLocalCoordinates(
            self.S
        )
        res = least_squares(
            self.costFun,
            self.optimVars,
            self.costFunJac,
            max_nfev=numIter,
            verbose=verbose,
        )
        return res

    def registerCallback(self, callback):
        self.callback = callback

    def getPosition(self, S):
        """Placeholder for child class."""
        raise NotImplementedError(
            "Returns the positions of the DLO at the given sample points."
        )

    def writeParametersToJson(self, savePath, fileName):
        """Placeholder for child class."""
        raise NotImplementedError("Saves the parameters of this models to a json file.")
