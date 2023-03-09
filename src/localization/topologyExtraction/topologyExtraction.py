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


class TopologyExtraction(topologyModel):
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

    def __init__(self, X, featureMatrix=None, *args, **kwargs):

        if type(X) is not np.ndarray or X.ndim != 2:
            raise ValueError("X must be at a 2D numpy array.")
        if X.shape[0] < X.shape[1]:
            raise ValueError("C must be a  mus be square.")

        if featureMatrix is not None:
            if type(featureMatrix) is not np.ndarray or featureMatrix.ndim != 2:
                raise ValueError("feature matrix must be at a 2D numpy array.")
            if (
                featureMatrix.shape[0] != featureMatrix.shape[1]
                or featureMatrix.shape[0] != X.shape[0]
            ):
                raise ValueError(
                    "The feature matrix must be square and same have the same length as X. Instead got {}".format(
                        featureMatrix.shape[0]
                    )
                )

        self.X = X
        self.featureMatrix = (
            distance_matrix(X, X) if featureMatrix is None else featureMatrix
        )
        adjacencyMatrix = self.findMinimalSpanningTree(self.featureMatrix)
        super().__init__(adjacenyMatrix=adjacencyMatrix, *args, **kwargs)

    def findMinimalSpanningTree(self, featureMatrix):
        """Returns the minimal spanning tree betwwen the points given in X

        Args:
            featureMatrix(np.array): feature Matrix containing the cost between all nodes

        Returns:
            symmetricAdjacencyMatrix(csgraph): symmetric adjacencyMatrix
        """
        adjacencyMatrix = minimum_spanning_tree(featureMatrix)
        symmetricAdjacencyMatrix = (
            adjacencyMatrix.toarray().astype(float)
            + adjacencyMatrix.toarray().astype(float).T
        )
        return symmetricAdjacencyMatrix

    def getAdjacentPointPairs(self):
        pointPairs = []
        for edge in self.getEdges():
            nodePair = edge.getNodes()
            thisNodeIdx = self.getNodeIndex(nodePair[0])
            otherNodeIdx = self.getNodeIndex(nodePair[1])
            pointPairs.append((self.X[thisNodeIdx, :], self.X[otherNodeIdx, :]))
        return pointPairs
