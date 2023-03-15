import sys, os
import numpy as np
from scipy.spatial import distance_matrix
import numbers
from warnings import warn

try:
    sys.path.append(os.getcwd().replace("/src/localization/downsampling", ""))
    from src.utils.utils import gaussian_kernel, knn
    from src.localization.downsampling.downsampling import (
        Downsampling,
    )
except:
    print("Imports for Data Reduction failed.")
    raise


class DataReduction(Downsampling):
    """Base class for dimensionality reduction methods

    Attributes:
    -------------
    X: numpy array
        NxD array of input points

    Y: numpy array
        MxD array of source points (e.g. 3D point cloud)

    T: numpy array
        NxD array of target points (transformed source points)

    N: int
        Number of source points

    M: int
        Number of data points

    D: int
        Dimensionality of source points

    iterations: int
        The current iteration throughout the registration

    max_iterations: int
        Maximum number of iterations the registration performs before terminating
    """

    def __init__(self, X=None, max_iterations=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if X is not None:
            if type(X) is not np.ndarray or X.ndim != 2:
                raise ValueError(
                    "The source point cloud (X) must be at a 2D numpy array."
                )

            if X.shape[1] != self.Y.shape[1]:
                raise ValueError(
                    "Both point clouds need to have the same number of dimensions."
                )
            if X.shape[0] < X.shape[1]:
                raise ValueError(
                    "The dimensionality is larger than the number of points. Possibly the wrong orientation of X."
                )
        if max_iterations is not None and (
            not isinstance(max_iterations, numbers.Number) or max_iterations < 0
        ):
            raise ValueError(
                "Expected a positive integer for max_iterations instead got: {}".format(
                    max_iterations
                )
            )
        elif isinstance(max_iterations, numbers.Number) and not isinstance(
            max_iterations, int
        ):
            warn(
                "Received a non-integer value for max_iterations: {}. Casting to integer.".format(
                    max_iterations
                )
            )
            max_iterations = int(max_iterations)
        if X is not None:
            self.X = X
            self.T = X
            (self.N, self.D) = self.X.shape
        self.max_iterations = 100 if max_iterations is None else max_iterations
        self.iteration = 0
