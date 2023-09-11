import sys
import os
import numpy as np
from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import johnson

try:
    sys.path.append(
        os.getcwd().replace("/src/localization/minmalSpanningTreeExtraction", "")
    )
    from src.utils.utils import minimalSpanningTree
    from src.localization.topologyExtraction.minimalSpanningTreeTopology import (
        MinimalSpanningTreeTopology,
    )
except:
    print("Imports for Minimal-Spanning-Tree Extraction failed.")
    raise


class MinimalSpanningTreeExtraction(object):
    def __init__(self, X=None, nPaths=None, *args, **kwargs):
        """Extracts a tree from a point set consisting of the N longest paths on the minimal spanning tree of the given point set.

        Args:
            X (np.array):
                point set
            nPaths: number of longest paths in the point cloud which compose the minimal spanning tree topology.
        """
        if X is not None:
            if type(X) is not np.ndarray or X.ndim != 2:
                raise ValueError(
                    "The source point cloud (X) must be at a 2D numpy array."
                )

            if X.shape[0] < X.shape[1]:
                raise ValueError(
                    "The dimensionality is larger than the number of points. Possibly the wrong orientation of X."
                )
        self.X = X
        self.nPaths = nPaths

    def get_max_indices_sorted(self, matrix):
        # Flatten the matrix
        flattened = matrix.flatten()

        # Get indices that would sort the flattened matrix
        sorted_indices = np.argsort(flattened)

        # Reverse the result to get indices in descending order of value
        indices_of_max_values_flat = sorted_indices[::-1]

        # Convert flattened indices back to matrix indices
        matrix_indices = np.column_stack(
            np.unravel_index(indices_of_max_values_flat, matrix.shape)
        )
        return matrix_indices

    def findNLongestPaths(self, adjacencyMatrix, nLongestPaths):
        longest_paths = []
        current_graph = adjacencyMatrix.copy()
        for n in range(0, nLongestPaths):
            dist_matrix, predecessors = johnson(
                csgraph=current_graph, directed=False, return_predecessors=True
            )
            dist_matrix[np.isinf(dist_matrix)] = 0
            if len(longest_paths) >= 1:
                # find the longest path connected to the already found paths
                sorted_maxima_indices = self.get_max_indices_sorted(dist_matrix)
                foundConnectedPath = False
                for indices in sorted_maxima_indices:
                    path = [indices[1]]
                    predecessor = path[0]
                    while predecessor != indices[0]:
                        predecessor = predecessors[indices[0], predecessor]
                        path.append(predecessor)
                    for foundPath in longest_paths:
                        newPathStartNodes = set(indices)
                        alreadyFoundNodes = set(foundPath)
                        if len(newPathStartNodes.intersection(alreadyFoundNodes)) > 0:
                            foundConnectedPath = True
                            break
                        else:
                            foundConnectedPath = False
                    if foundConnectedPath:
                        break
            else:
                # find the longest path
                max_distant_point_idxs = np.unravel_index(
                    np.argmax(dist_matrix, axis=None), dist_matrix.shape
                )
                path = [max_distant_point_idxs[1]]
                predecessor = path[0]
                while predecessor != max_distant_point_idxs[0]:
                    predecessor = predecessors[max_distant_point_idxs[0], predecessor]
                    path.append(predecessor)
                # for row_idx in path:
                #     for col_idx in path:
                #         current_graph[row_idx, col_idx] = 0
            # remove the found longest path from the graph
            predecessorIdxs = path[1:]
            nodeIdxs = path[:-1]
            for nodeIdx, predecessorIdx in zip(nodeIdxs, predecessorIdxs):
                current_graph[nodeIdx, predecessorIdx] = 0
                current_graph[predecessorIdx, nodeIdx] = 0
            longest_paths.append(path)
        return longest_paths

    def extractTopology(self, X=None, nPaths=None):
        X = self.X if X is None else X
        nPaths = self.nPaths if nPaths is None else nPaths

        minSpanTreeAdjMatrix = minimalSpanningTree(distance_matrix(X, X))
        longestPaths = self.findNLongestPaths(minSpanTreeAdjMatrix, nPaths)
        # l = [item for sublist in longestPaths for item in sublist]
        remainingPointsIdxs = list(
            set(item for sublist in longestPaths for item in sublist)
        )
        remainingPointsIdxs.sort()
        connectivityMatrix = np.zeros((len(X), len(X)))
        for path in longestPaths:
            predecessorIdxs = path[1:]
            nodeIdxs = path[:-1]
            for nodeIdx, predecessorIdx in zip(nodeIdxs, predecessorIdxs):
                connectivityMatrix[nodeIdx, predecessorIdx] = 1
                connectivityMatrix[predecessorIdx, nodeIdx] = 1
        reducedAdjacencyMatrix = np.zeros(
            (len(remainingPointsIdxs), len(remainingPointsIdxs))
        )
        for i, oldRowIdx in enumerate(remainingPointsIdxs):
            for j, oldColIdx in enumerate(remainingPointsIdxs):
                reducedAdjacencyMatrix[i, j] = (
                    connectivityMatrix[oldRowIdx, oldColIdx]
                    * minSpanTreeAdjMatrix[oldRowIdx, oldColIdx]
                )
        reducedPointSet = X[remainingPointsIdxs, :]
        extractedTopology = MinimalSpanningTreeTopology(
            X=reducedPointSet, featureMatrix=reducedAdjacencyMatrix
        )
        return extractedTopology
