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
except:
    print("Imports for discrete shape reconstruction failed.")
    raise


class BDLOReconstruction(ShapeReconstruction):
    """Base class for reconstructing the shape of BDLOs
    Reconstuction aims to obtain the parameters describing the shape of a BDLO from a (high dimensional) set of points (e.g. a point cloud) in cartesian space corresponding to the current configuration of the BDLO.
    This class  assumes the correspondance problem is solved such that for every point in cartesian space, the corresponding location on the BDLO is known.

    Attributes:
    -------------
    CBY (list of dict):
        branch correspondance information for each point in Y such that CBY[i] is the branch of the DLO the i-th point in Y is corresponding to.

    K (int):
        number of branches
    """

    def __init__(
        self,
        bdlo: BranchedDeformableLinearObject,
        CBY,
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
        self.bdlo = bdlo
        self.q = self.bdlo.skel.getPositions()
        self.K = len(CBY)
        self.CBY = CBY
        self.X = np.zeros((self.M, self.D))
        for i in range(0, self.M):
            self.X[i] = self.bdlo.getCartesianPositionFromBranchLocalCoordinate(
                self.CBY[i], self.SY[i]
            )
        # lockedDofs=[3, 4, 5]
        unlockedDofs = []
        for i, branch in enumerate(self.bdlo.getBranches()):
            branchRootDofIndices = self.bdlo.getBranchRootDofIndices(i)
            for branchRootDofIndex in branchRootDofIndices:
                unlockedDofs.append(branchRootDofIndex)
        lockedDofs = list(range(0, self.bdlo.skel.getNumDofs()))
        for dof in lockedDofs:
            if dof in unlockedDofs:
                lockedDofs.remove(dof)
        self.optimVars, self.mappingDict = self.initOptimVars(lockedDofs)

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
        for i in range(0, self.M):
            self.X[i] = self.bdlo.getCartesianPositionFromBranchLocalCoordinate(
                self.CBY[i], self.SY[i]
            )

        # update Iteration Number
        self.iter += 1

        if callable(self.callback):
            self.callback()

    def costFun(self, optimVars):
        self.updateParameters(optimVars)
        # for i, y in enumerate(self.Y):
        #     correspondingBranchIndex = self.CBY[i]
        #     correspondingLocalCoordinate = self.SY[i]
        #     x = self.bdlo.getCartesianPositionFromBranchLocalCoordinate(
        #         correspondingBranchIndex, correspondingLocalCoordinate
        #     )
        #     errors[i] = np.linalg.norm(y - x)
        #     self.X[i] = x
        error = self.wPosDiff * np.sum(
            np.square(np.linalg.norm(self.Y - self.X, axis=1))
        )
        return error

    def costFunJac(self, optimVars):
        # determine dofs from optimVars
        self.updateParameters(optimVars)

        # determineJacobians
        jacobians = []
        for i in range(0, self.M):
            jacobians.append(
                self.bdlo.getJacobianFromBranchLocalCoordinate(self.CBY[i], self.SY[i])
            )

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
                    paddedJacobian = jacobian.copy()
                else:
                    raise ValueError("Jacobian seems to have wrong dimension.")
                jacobianRows.append(paddedJacobian[3:6, correspondingDartIndex])
            jacobianMultiplikatorMatrix = np.vstack(jacobianRows)
            # calcualte cost function derivative

            J[i] = np.sum(
                -2
                * self.wPosDiff
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
        return res

    def getPosition(self, S):
        """Placeholder for child class."""
        raise NotImplementedError(
            "Returns the positions of the DLO at the given sample points."
        )

    def writeParametersToJson(self, savePath, fileName):
        """Placeholder for child class."""
        raise NotImplementedError("Saves the parameters of this models to a json file.")
