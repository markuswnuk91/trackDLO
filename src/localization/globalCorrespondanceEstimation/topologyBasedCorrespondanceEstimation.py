import sys
import os
import numpy as np
import numbers
from warnings import warn

try:
    sys.path.append(os.getcwd().replace("/src/localization", ""))
    from src.localization.topologyExtraction import TopologyExtraction
except:
    print("Imports for Topology-based Correspondance Estimation failed.")
    raise


class TopologyBasedCorrespondanceEstimation(TopologyExtraction):
    """

    Attributes:
    X: numpy array
        NxD array representing a semi structured, well aligned reduced representation of the source points

    Y: numpy array
        MxD array of unordered, unstructured data points

    topologyTemplate: topologyModel
        topologyModel representing the BDLO for which the correspondances should be estimated

    N: int
        Number of points in the reduced representation

    M: int
        Number of data points

    D: int
        Dimensionality of source and data points
    """

    def __init__(self, Y, templateTopology, *args, **kwargs):
        if type(Y) is not np.ndarray or Y.ndim != 2:
            raise ValueError("The source point cloud (Y) must be a 2D numpy array.")

        if Y.shape[0] < Y.shape[1]:
            raise ValueError(
                "The dimensionality is larger than the number of points. Possibly the wrong orientation of Y."
            )

        self.Y = Y
        self.templateTopology = templateTopology
        self.extractedTopology = self.extractTopology(self.Y)
