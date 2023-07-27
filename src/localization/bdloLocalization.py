import os
import sys
import numpy as np
import numbers
from warnings import warn
from scipy.optimize import least_squares
import time

try:
    sys.path.append(os.getcwd().replace("/src/reconstruction", ""))
    from src.reconstruction.shapeReconstruction import ShapeReconstruction
    from src.simulation.bdlo import BranchedDeformableLinearObject
    from src.localization.correspondanceEstimation.topologyBasedCorrespondanceEstimation import (
        TopologyBasedCorrespondanceEstimation,
    )
    from src.utils.utils import dampedPseudoInverse
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

    def __init__(self, S, callback=None, jacobianDamping=None, *args, **kwargs):
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
        self.XLog = []
        self.qLog = []
        (self.N, _) = self.X.shape
        self.jacobianDamping = 1 if jacobianDamping is None else jacobianDamping
        # unlockedDofs = []
        # for i, branch in enumerate(self.bdlo.getBranches()):
        #     branchRootDofIndices = self.bdlo.getBranchRootDofIndices(i)
        #     for branchRootDofIndex in branchRootDofIndices:
        #         unlockedDofs.append(branchRootDofIndex)
        # lockedDofs = list(range(0, self.bdlo.skel.getNumDofs()))
        # for dof in lockedDofs:
        #     if dof in unlockedDofs:
        #         lockedDofs.remove(dof)
        # lockedDofs = [3, 4, 5]
        lockedDofs = None
        self.optimVars, self.mappingDict = self.initOptimVars()
        # self.optimVars, self.mappingDict = self.initOptimVars(None)
        self.runTimes["inverseKinematicsIterations"] = []

    def sampleCartesianPositionsFromModel(self, S=None):
        X = np.zeros((self.Ns * self.K, self.D))
        for b in range(0, self.K):
            if b == 0:
                X[b * self.Ns : (b + 1) * self.Ns] = np.flipud(
                    self.bdlo.getCartesianPositionsFromBranchLocalCoordinates(b, self.S)
                )
            else:
                X[
                    b * self.Ns : (b + 1) * self.Ns
                ] = self.bdlo.getCartesianPositionsFromBranchLocalCoordinates(b, self.S)
        return X

    def getModelPositionAndJacobian(self, b, s, q):
        self.bdlo.skel.setPositions(q)
        x = self.bdlo.getCartesianPositionFromBranchLocalCoordinate(b, s)
        jacobian = self.bdlo.getJacobianFromBranchLocalCoordinate(b, s)
        return (x, jacobian)

    def forwardKinematics(self, q, S):
        X = np.zeros((self.K * len(self.S), 3))
        self.bdlo.skel.setPositions(q)
        for k in range(self.bdlo.getNumBranches()):
            X[
                k * len(self.S) : (k + 1) * len(self.S)
            ] = self.bdlo.getCartesianPositionsFromBranchLocalCoordinates(k, S)
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
        # self.q[3:6] = (np.linalg.inv(self.C) @ self.YTarget)[-1, :]
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
            np.square(np.linalg.norm(self.C.T @ self.YTarget - self.X, axis=1))
        )
        return error

    def costFunJac(self, optimVars):
        # determine dofs from optimVars
        self.updateParameters(optimVars)

        # determineJacobians
        jacobians = []
        for b in range(0, self.K):
            if b == 0:
                jacobians_backwards_ordered = []
                for s in self.S:
                    jacobians_backwards_ordered.append(
                        self.bdlo.getJacobianFromBranchLocalCoordinate(b, s)
                    )
                for jacobian in reversed(jacobians_backwards_ordered):
                    jacobians.append(jacobian)
            else:
                for s in self.S:
                    jacobians.append(
                        self.bdlo.getJacobianFromBranchLocalCoordinate(b, s)
                    )

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
                    (self.C.T @ self.YTarget - self.X) * (jacobianMultiplikatorMatrix),
                    axis=1,
                )
            )
        # update Iteration Number
        self.iter += 1

        # measure runtime
        runtimeInverseKinematicsIteration_end = time.time()
        self.currentInverseKinematicsTimeStamp = time.time
        if self.iter == 1:
            runtimePerIteration = (
                runtimeInverseKinematicsIteration_end
                - self.runtimeInverseKinematics_start
            )
        else:
            runtimePerIteration = (
                runtimeInverseKinematicsIteration_end
                - self.currentInverseKinematicsTimeStamp
            )
        self.runTimes["inverseKinematicsIterations"].append(runtimePerIteration)

        # save X and q
        self.XLog.append(self.X)
        self.qLog.append(self.q)
        if callable(self.callback):
            self.callback()
        return J

    def reconstructShape(self, numIter: int = -1, verbose=0, method="least_squares"):
        runTimeLocalization_start = time.time()
        if self.extractedTopology is None:
            warn("No topology yet extracted. Extracting topology ...")
            self.extractTopology()
        (self.YTarget, self.X, self.C) = self.findCorrespondancesFromLocalCoordinates(
            self.S
        )
        if method == "least_squares":
            self.runtimeInverseKinematics_start = time.time()
            if numIter == -1:
                numIter = None
            res = least_squares(
                self.costFun,
                self.optimVars,
                self.costFunJac,
                max_nfev=numIter,
                verbose=verbose,
            )
            q = res.x
        elif method == "IK":
            runtimeInverseKinematics_start = time.time()
            if numIter == -1:
                numIter = 100
            self.X_desired = self.C.T @ self.YTarget
            # kinematic regularization (solve IK iteratively)
            dq = np.zeros(self.bdlo.skel.getNumDofs())
            q = self.q
            ik_iterations = numIter
            for i in range(0, ik_iterations):
                runTimeInverseKinematicsIteration_start = time.time()
                jacobians = []
                X_error = []
                X_current = []
                for b in range(0, self.K):
                    for s in self.S:
                        x_current, J_hat = self.getModelPositionAndJacobian(
                            b, s, self.q
                        )
                        X_current.append(x_current)
                        jacobians.append(J_hat[3:6, :])
                self.X = np.array(X_current)
                X_error = self.X_desired - self.X
                J = np.vstack(jacobians)
                dq = dampedPseudoInverse(J, self.jacobianDamping) @ X_error.flatten()
                self.q = self.q + dq
                # save X and q
                self.XLog.append(self.X)
                self.qLog.append(self.q)
                if callable(self.callback):
                    self.callback()

                runTimeInverseKinematicsIteration_end = time.time()
                self.runTimes["inverseKinematicsIterations"].append(
                    runTimeInverseKinematicsIteration_end
                    - runTimeInverseKinematicsIteration_start
                )
            self.X = self.forwardKinematics(self.q, self.S)

        runtimeInverseKinematics_end = time.time()
        self.runTimes["inverseKinematics"] = (
            runtimeInverseKinematics_end - runtimeInverseKinematics_start
        )
        self.runTimes["localization"] = (
            runtimeInverseKinematics_end - runTimeLocalization_start
        )
        return self.q

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
