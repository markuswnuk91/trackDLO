import sys
import os
import numpy as np
import numbers
from warnings import warn
from scipy.spatial import distance_matrix

try:
    sys.path.append(os.getcwd().replace("/src/localization/topologyExtraction", ""))
    from src.modelling.topologyModel import topologyModel
    from src.utils.utils import minimalSpanningTree
except:
    print("Imports for Minimal-Spanning-Tree failed.")
    raise


class MinimalSpanningTreeTopology(topologyModel):
    def __init__(self, X, featureMatrix=None, *args, **kwargs):
        """Builds a minimimum spanning tree based topology model between the points given in X based on the provided featureMatrix, where the featureMatrix provides the cost of travelling between two adjacent points (e.g. the default feature is euclidean distance between the points given in X)
        The minimum spanning tree searches a minimum cost path through the featureMatrix to connect all its nodes.

        Args:
            X (np.array):
                coordinates cooresponding to the nodes of the feature matrix
            featureMatrix (np.array, optional):
                feature matrix of costs between the nodes. Defaults to euclidean distance between the points in X.
        """
        if type(X) is not np.ndarray or X.ndim != 2:
            raise ValueError("The source point cloud (X) must be at a 2D numpy array.")

        if X.shape[0] < X.shape[1]:
            raise ValueError(
                "The dimensionality is larger than the number of points. Possibly the wrong orientation of X."
            )
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
        if featureMatrix is None:
            self.featureMatrix = distance_matrix(self.X, self.X)
        else:
            self.featureMatrix = featureMatrix
        self.adjacencyMatrix = self.findMinimalSpanningTree(self.featureMatrix)
        super().__init__(adjacencyMatrix=self.adjacencyMatrix, *args, **kwargs)

    def findMinimalSpanningTree(self, featureMatrix):
        return minimalSpanningTree(featureMatrix)

    def getAdjacentPointPairs(self):
        pointPairs = []
        for edge in self.getEdges():
            nodePair = edge.getNodes()
            thisNodeIdx = self.getNodeIndex(nodePair[0])
            otherNodeIdx = self.getNodeIndex(nodePair[1])
            pointPairs.append((self.X[thisNodeIdx, :], self.X[otherNodeIdx, :]))
        return pointPairs
