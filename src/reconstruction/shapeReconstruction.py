import numpy as np
import numbers
from warnings import warn


class ShapeReconstruction(object):
    """Base class for reconstructing the shape of DLOs
    Reconstuction aims to obtain the parameters describing the shape of a DLO from a (high dimensional) set of points (e.g. a point cloud) in cartesian space corresponding to the current configuration of the DLO.
    This class  assumes the correspondance problem is solved such that for every point in cartesian space, the corresponding location on the DLO is known.

    Attributes:
    -------------
    Y: numpy array
        MxD array of target points

    M: int
        Number of data points

    SY: numpy array
       Mx1 array of normalized local coodinates in [0,1] the target points in Y correspond to.

    D: int
        Dimensionality of source points and weights
    """

    def __init__(self, Y, SY, callback=lambda **kwargs: None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if type(Y) is not np.ndarray or Y.ndim != 2:
            raise ValueError("The target points (Y) must be at a 2D numpy array.")

        if Y.shape[0] < Y.shape[1]:
            raise ValueError(
                "The dimensionality is larger than the number of points. Possibly wrong orientation of X."
            )

        if np.any(SY > 1) or np.any(SY < 0):
            raise ValueError(
                "Obtained non-nomalized coordinates for local coordinates. Expected values to be normalized by the length of the DLO in [0,1]"
            )
        self.Y = Y
        self.SY = SY
        (self.M, self.D) = self.Y.shape
        self.callback = None if callback is None else callback

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
