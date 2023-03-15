import numpy as np
from scipy.spatial import distance_matrix
import numbers
from warnings import warn


class Downsampling(object):
    """Base class for dimensionality reduction methods

    Attributes:
    -------------
    Y: numpy array
        MxD array of source points (e.g. 3D point cloud)

    T: numpy array
        NxD array of target points (transformed source points)

    M: int
        Number of data points

    D: int
        Dimensionality of source points
    """

    def __init__(self, Y=None, *args, **kwargs):
        if Y is not None:
            if type(Y) is not np.ndarray or Y.ndim != 2:
                raise ValueError("The target point cloud (Y) must be a 2D numpy array.")

            if Y.shape[0] < Y.shape[1]:
                raise ValueError(
                    "The dimensionality is larger than the number of points. Possibly the wrong orientation of Y."
                )
            self.Y = Y
            (self.M, self.D) = self.Y.shape
        else:
            self.Y = None
        self.callback = None

    def calculateReducedRepresentation(self):
        """
        Placeholder for child classes.
        """
        raise NotImplementedError(
            "Method for calculating the reduced representation should be defined in child classes."
        )

    def getCorrespondences(self):
        """Returns the correspondences of the reduced point set T to the source points Y

        Returns:
        C (list(np.array)): array of correspondances such that Y[C[0],:] are the points corresponding to T[0,:]
        """
        C = []
        distances = distance_matrix(self.T, self.Y)
        correspondingIndices = np.argmin(distances, axis=0)
        for i in range(0, self.N):
            C.append(np.where(correspondingIndices == i)[0])
        return C

    def getCorrespondingPoints(self, i):
        """Returns the corresponding source points in Y for a target point in T.

        Input
            i (int):
                index of the reduced point T for which the corresponding points should be returned.
        Returns
            YCorresponding: re
        """
        C = self.getCorrespondences()
        return self.Y[C[i]]

    def registerCallback(self, callback):
        self.callback = callback
