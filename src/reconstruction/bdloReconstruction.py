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


class bdloReconstruction(ShapeReconstruction):
    """Base class for reconstructing the shape of BDLOs
    Reconstuction aims to obtain the parameters describing the shape of a BDLO from a (high dimensional) set of points (e.g. a point cloud) in cartesian space corresponding to the current configuration of the BDLO.
    This class  assumes the correspondance problem is solved such that for every point in cartesian space, the corresponding location on the BDLO is known.

    Attributes:
    -------------
    BC (list of dict):
        branch correspondance information as a list of dict. For each branch contains a dict that assigning each point in Y and each local coordinate in SY to a branch, such that Y[BC[0]["Y"],:] are the points in Y corresponding to branch 0 and SY[BC[0]["S"]] are the local coordinates correspoding to these points.

    K (int):
        number of branches
    """

    def __init__(
        self,
        bdlo: BranchedDeformableLinearObject,
        BC,
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
        self.bdlo = bdlo
        self.K = len(BC)
        self.BC = BC
        self.X = np.zeros(self.M, self.D)
        for i in range(0, self.K):
            self.X[BC[i], :] = self.bdlo.getCaresianPositionsFromLocalCoordinates(
                branchIndex=i, S=self.SY[BC[i]]
            )

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
        # update skeleton
        self.setPositions(self.q)

        # determine Positions
        for i in range(0, self.K):
            self.X = self.bdlo.getCaresianPositionsFromLocalCoordinates(self.SY)

        # update Iteration Number
        self.iter += 1

        if callable(self.callback):
            self.callback()

    def getPosition(self, S):
        """Placeholder for child class."""
        raise NotImplementedError(
            "Returns the positions of the DLO at the given sample points."
        )

    def writeParametersToJson(self, savePath, fileName):
        """Placeholder for child class."""
        raise NotImplementedError("Saves the parameters of this models to a json file.")
