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
        (self.N, self.D) = self.X.shape
        if featureMatrix is None:
            self.featureMatrix = distance_matrix(self.X, self.X)
        else:
            self.featureMatrix = featureMatrix
        self.adjacencyMatrix = self.findMinimalSpanningTree(self.featureMatrix)
        super().__init__(adjacencyMatrix=self.adjacencyMatrix, *args, **kwargs)
        for i, node in enumerate(self.nodes):
            node.addNodeInfo(key="cartesianPosition", info=self.X[i])

    def findMinimalSpanningTree(self, featureMatrix):
        return minimalSpanningTree(featureMatrix)

    def getCorrespondingBranchFromNode(self, nodeIndex: int):
        """returns the corresponding branch index for a node. returns only a single branch index. If the node is a branch node and corresponds to multiple branches, only the correspondance to the first branch in the branchlist is returned.
        Args:
            nodeIndex (int): index of the node for which the branch correspondance should be determined

        Returns:
            corresponingBranchIndex (int): index of the branch the node corresponds to
        """
        correspondingBranches = self.getBranchesFromNode(self.getNodes()[nodeIndex])
        if len(correspondingBranches) <= 0:
            raise ValueError(
                "Node does not correspond to any branch. Something went wrong."
            )
        return self.getBranchIndex(correspondingBranches[0])

    def calculateCorrespondingNodesForPointSet(self, pointSet):
        """returns the index of the closest nodes for each point in the given point set.
        Uses the euclidean distance between the point in the set and the node postions X to calculcate the correspondacen.

        Args:
            pointSet (np.array): point set of size MxD for which the node with closest cartesian distance should be determined

        Returns:
            CN: corresponding nodes for each point in the set, such that CN[i] is the nodeIndex corresponding to point i of the point set.
        """
        CN = []
        distances = distance_matrix(pointSet, self.X)
        correspondingIndices = np.argmin(distances, axis=0)
        for i in range(0, pointSet.shape[0]):
            CN.append(np.where(correspondingIndices == i)[0][0])
        return CN

    def calculateNodeCorrespondanceFromPointSet(self, pointSet):
        """returns the correspondances between the given point set and the nodes of this topology representation

        Args:
            pointSet (np.array): point set of size MxD for which the node with closest cartesian distance should be determined

        Returns:
            NC (list): list of arrays of corresponancees each node, such that pointSet[NC[i],:] gives the points corresponding to node i
        """
        NC = []
        distances = distance_matrix(self.X, pointSet)
        correspondingIndices = np.argmin(distances, axis=0)
        for i in range(0, len(self.nodes)):
            NC.append(np.where(correspondingIndices == i)[0])
        return NC

    def calculateCorrespondingBranchesForPointSet(self, pointSet):
        """returns the corresponding branches for each point in the given point set

        Args:
            pointSet (np.array):
                point set for which the branch correspondances should be determined

        Returns
            CB(np.array):
                array of corresponding branch indices for each point in the set, such that CB[i] is the branchIndex corresponding to point i of the point set.
        """
        CB = []
        correspondingNodeIndices = self.calculateCorrespondingNodesForPointSet(pointSet)
        for i, point in enumerate(pointSet):
            correspondingNodeIndex = correspondingNodeIndices[i]
            correspondingBranchIndex = self.getCorrespondingBranchFromNode(
                correspondingNodeIndex
            )
            CB.append(correspondingBranchIndex)
        return CB

    def calculateBranchCorrespondanceForPointSet(self, pointSet):
        """returns the correspondances between the given point set and the branches of this topology representation

        Args:
            pointSet (np.array):
                point set for which the branch correspondances should be determined

        Returns
            BC(list of np.array):
                list of arrays of correspondences for each branch, such that pointSet[BC[i],:] gives the points corresponding to branch i
        """
        BC = []
        nodeCorrespondances = self.calculateNodeCorrespondanceFromPointSet(pointSet)

        raise notImplementedError

    def getAdjacentPointPairs(self):
        pointPairs = []
        for edge in self.getEdges():
            nodePair = edge.getNodes()
            thisNodeIdx = self.getNodeIndex(nodePair[0])
            otherNodeIdx = self.getNodeIndex(nodePair[1])
            pointPairs.append((self.X[thisNodeIdx, :], self.X[otherNodeIdx, :]))
        return pointPairs

    def getAdjacentPointPairsAndBranchCorrespondance(self):
        pointPairs = []
        for edge in self.getEdges():
            nodePair = edge.getNodes()
            thisNodeIdx = self.getNodeIndex(nodePair[0])
            otherNodeIdx = self.getNodeIndex(nodePair[1])
            correspondingBranchIdx = self.getBranchIndexFromEdge(edge)
            pointPairs.append(
                (
                    self.X[thisNodeIdx, :],
                    self.X[otherNodeIdx, :],
                    correspondingBranchIdx,
                )
            )
        return pointPairs
