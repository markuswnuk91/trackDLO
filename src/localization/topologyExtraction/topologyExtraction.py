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
        NxD array of source points

    N: int
        Number of source points

    D: int
        Dimensionality of source and target points
    """

    def __init__(self, X, *args, **kwargs):
        if type(X) is not np.ndarray or X.ndim != 2:
            raise ValueError("The source point cloud (X) must be at a 2D numpy array.")
        if X.shape[0] < X.shape[1]:

            raise ValueError(
                "The dimensionality is larger than the number of points. Possibly the wrong orientation of X and Y."
            )
        self.X = X
        (self.N, self.D) = self.X.shape
        adjacencyMatrix = self.findMinimalSpanningTree(self.X)
        super().__init__(adjacencyMatrix=adjacencyMatrix, *args, **kwargs)

    def findMinimalSpanningTree(self, X):
        """Returns the minimal spanning tree betwwen the points given in X

        Args:
            X (np.array): Points over which the minimum spanning tree shoud be found.

        Returns:
            symmetricAdjacencyMatrix(csgraph): symmetric adjacencyMatrix
        """
        adjacencyMatrix = minimum_spanning_tree(distance_matrix(X, X))
        symmetricAdjacencyMatrix = (
            adjacencyMatrix.toarray().astype(float)
            + adjacencyMatrix.toarray().astype(float).T
        )
        return symmetricAdjacencyMatrix

    def extractTopologyRepresentation(self):
        adjacencyMatrix = self.findMinimalSpanningTree(self.X)
        topology = topologyModel(adjacencyMatrix)
        return topology

    def getAdjacentPointPairs(self):
        pointPairs = []
        for edge in self.getEdges():
            nodePair = edge.getNodes()
            thisNodeIdx = self.getNodeIndex(nodePair[0])
            otherNodeIdx = self.getNodeIndex(nodePair[1])
            pointPairs.append((self.X[thisNodeIdx, :], self.X[otherNodeIdx, :]))
        return pointPairs
