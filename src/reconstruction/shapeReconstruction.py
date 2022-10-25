import numpy as np
import numbers
from warnings import warn


class ShapeReconstruction(object):
    """Base class for reconstructing the shape of DLOs
    Reconstuction aims to obtain the continous shape of a DLO from a discrete representation (such as a point cloud or registration approach)

    Attributes:
    -------------
    Y: numpy array
        MxD array of target points (e.g. from a registration result)

    M: int
        Number of data points

    numS: int
        Number of collocation points along the DLO used to sample the continous shape

    Sc: numpy array
       numSx1 array of local coodinates of collocation points along the DLO used to sample the continous shape.

    Sx: numpy array
       Mx1 array of local coodinates the source points in X correspond to.

    D: int
        Dimensionality of source points and weights
    """

    def __init__(
        self, Y, Sx, L, numSc=None, callback=lambda **kwargs: None, *args, **kwargs
    ):
        if type(Y) is not np.ndarray or Y.ndim != 2:
            raise ValueError("The target points (Y) must be at a 2D numpy array.")

        if Y.shape[0] < Y.shape[1]:
            raise ValueError(
                "The dimensionality is larger than the number of points. Possibly wrong orientation of X."
            )

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

        if L is not None and (not isinstance(L, numbers.Number) or L < 0):
            raise ValueError(
                "Expected a positive float for length of the DLO. Instead got: {}".format(
                    L
                )
            )
        self.Y = Y
        self.Sx = Sx
        self.L = L
        (self.M, self.D) = self.Y.shape
        self.numSc = 100 if numSc is None else numSc
        self.Sc = np.linspace(0, self.L, self.numSc)
        self.callback = None if callback is None else callback

    def evalAnsatzFuns(self, S):
        """Placeholder for child class."""
        raise NotImplementedError(
            "Evalues the ansatz functions at the given sample points."
        )

    def getPosition(self, S):
        """Placeholder for child class."""
        raise NotImplementedError(
            "Returns the positions of the DLO at the given sample points."
        )
