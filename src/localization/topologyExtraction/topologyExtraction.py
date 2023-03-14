import sys
import os
import numpy as np
import numbers
from warnings import warn
from scipy.sparse.csgraph import minimum_spanning_tree, shortest_path
from scipy.spatial import distance_matrix

try:
    sys.path.append(os.getcwd().replace("/src/localization", ""))
    from src.modelling.topologyModel import topologyModel
    from src.dimreduction.dimensionalityReduction import DimensionalityReduction
except:
    print("Imports for Topology Extraction failed.")
    raise


class TopologyExtraction(object):
    """

    Attributes:
    X: numpy array
        NxD array of source points for the graph reconstruction. Should be a reduced point set where points are unordered but well aligned

    Y: numpy array
        MxD array of unordered, unstructured data points from which the data points X have been extracted

    N: int
        Number of source points

    M: int
        Number of data points

    D: int
        Dimensionality of source and data points
    """

    def __init__(self, Y, *args, **kwargs):

        if type(Y) is not np.ndarray or Y.ndim != 2:
            raise ValueError("The source point cloud (Y) must be a 2D numpy array.")

        if Y.shape[0] < Y.shape[1]:
            raise ValueError(
                "The dimensionality is larger than the number of points. Possibly the wrong orientation of Y."
            )
        self.Y = Y
        self.topology = None

    def extractTopology():
        """
        Placeholder for child classes.
        """
        raise NotImplementedError(
            "Method to extract the topology should be defined in child classes."
        )
