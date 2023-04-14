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
    from src.localization.topologyExtraction import minimalSpanningTreeTopology
except:
    print("Imports for discrete shape reconstruction failed.")
    raise


class BDLOLocalization(ShapeReconstruction):
    """Calss for perfoming a localization for a BDLO from point cloud data
    Localization aims to obtain the generalized coordinates corresponding to the current configuration of the BDLO from a (high dimensional) set of points (point cloud).
    This class uses a discrete finite segment BDLO model.

    Attributes:
    -------------
    bdlo (BranchedDeformableLinearObject):
        Finite segment model of the branched deformable linear object which provides an interface to obatain the positions and jacobians corresponding to a local coordinate in each branch of the BDLO

    Sb (np.array):
        sampling locations along the branches

    C (np.ndarray):
        2D corresponance matrix which orders the points in X such that they correspond to the points in Y according to: C@X ~ Y

    K (int):
        number of branches
    """

    def __init__(
        self,
        bdlo: BranchedDeformableLinearObject,
        S,
        extractedTopology: minimalSpanningTreeTopology = None,
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
        self.S = S
        self.Ns = len(S)
        self.q = self.bdlo.skel.getPositions()
        self.K = self.bdlo.getNumBranches()
        self.X = np.zeros((self.Ns, self.D))
        self.X = self.sampleCartesianPositionsFromModel()

        # perform topology extraction
        if extractedTopology is None:
            raise NotImplementedError
            # implement topology extraction here.
        else:
            self.extractedTopology = extractedTopology
        # determine Correspondance Matrix
        # self.C = C

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
        X = np.zeros((self.Ns, self.D))
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
        #     correspondingLocalCoordinate = self.SY[i]
        #     x = self.bdlo.getCartesianPositionFromBranchLocalCoordinate(
        #         correspondingBranchIndex, correspondingLocalCoordinate
        #     )
        #     errors[i] = np.linalg.norm(y - x)
        #     self.X[i] = x
        error = self.wPosDiff * np.sum(
            np.square(np.linalg.norm(self.Y - self.C @ self.X, axis=1))
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
                * self.wPosDiff
                * np.sum(
                    (self.Y - self.C @ self.X) * self.C @ jacobianMultiplikatorMatrix,
                    axis=1,
                )
            )
        # update Iteration Number
        self.iter += 1

        if callable(self.callback):
            self.callback()
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
