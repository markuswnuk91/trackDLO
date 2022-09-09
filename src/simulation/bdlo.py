from operator import length_hint
import dartpy as dart
import numpy as np


class node:
    """
    Nodes are the basic elements of a graph.
    Each node contains the information to which parent it is connected.

    Attributes:
        parentNode: The parent of this node
        edge: The edge connecting this node with its paretent.
        Edges can contain additional infromation such as weights
    """

    ID = 0

    def __init__(self, parentNode=None, **kwargs):
        self.parentNode = parentNode
        if parentNode is None:
            self.edge = None
        self.edge = edge(parentNode, self, **kwargs)
        self.ID = node.ID
        node.ID += 1

    def getParent(self):
        return self.parentNode

    def getEdge(self):
        return self.edge

    def hasParentNode(self):
        return self.parentNode is not None


class edge:
    """
    Edges are the basic elements of a graph.
    Each edge connects two nodes.
    It holds additional information such as weigts.

    Attributes:
        parentNode (node): The parent node of this edge
        childNode (node): The child node of this edge.
        weights (dict): Weights can contain additional information in a dict.
    """

    def __init__(self, parentNode, childNode, **kwargs):
        self.parentNode = parentNode
        self.childNode = childNode
        self.weights = {}
        for weight in kwargs:
            self.weights[weight] = kwargs[weight]

    def getParentNode(self):
        return self.parentNode

    def getChildNode(self):
        return self.childNode

    def getWeight(self):
        return self.weights

    def setWeights(self, weigths):
        self.weights = weigths


class branch:
    """
    A branch is a collection of nodes.

    Attributes:
        startNode (node): The node the branch starts at.
        endNode (node): The node the branch ends at.
        memberNodes (list of nodes): Membernodes of the branch (exclusively the start and end node)
    """

    ID = 0

    def __init__(self, name, startNode, endNode, branchLength, radius):
        self.name = name
        self.startNode = startNode
        self.endNode = endNode
        self.memberNodes = []
        self.nodes = []
        self.edges = []
        self.branchLength = branchLength
        self.radius = radius
        self.ID = branch.ID
        branch.ID += 1

        self._collectMemberNodes()
        self.nodes.insert(0, startNode)
        self.nodes.append(endNode)
        self._collectEdges()
        self._initEdges()

    def getStartNode(self):
        return self.startNode

    def getEndNode(self):
        return self.endNode

    def getMemberNodes(self):
        return self.memberNodes

    def getNodes(self):
        return self.nodes

    def getBranchLength(self):
        return self.branchLength

    def getRadius(self):
        return self.radius

    def getParentBranch(self):
        return self.parent

    def hasParentBranch(self):
        return self.parent is not None

    def _collectMemberNodes(self):
        node = self.endNode
        while node is not self.startNode:
            if node is not None:
                node = node.getParent()
                self.memberNodes.append(node)
            else:
                raise ValueError("End node and start node seem to be not connected.")
        self.memberNodes.reverse()

    def _collectEdges(self):
        for node in self.nodes:
            self.edges.append(node.getEdge())

    def _initEdges(self):
        numNodes = len(self.nodes)
        weights = {"length": self.length / numNodes, "radius": self.radius}
        for edge in self.edges:
            edge.setWeights(weights)

    def __str__(self):
        _str = ""
        _str += "Branch ID: {}".format(self.ID)
        return _str


class BDLO:
    """
    Class implementing a discription for BDLOs based on a graph representation to interface with dart's skeleton class.
    A BDLO is a collection of branches.

    Attributes:
        name (str): name of the BDLO
        branches (list of branches): The branches the BDLO consists of.
        skel (dart.dynamics.Skeleton): dart skeleton used to simulate the BDLO.
    """

    ID = 0

    def __init__(self, name, skel) -> None:

        self.name = name
        self.skel = dart.dynamics.Skeleton(name=name)
        self.branches = []

    def addBranch(
        self,
        branchLength: float,
        radius: float,
        discretization: int,
        startNode: node = None,
        endNode: node = None,
    ):
        """
        Branches

        Args:
            startNode (node): starting node the branch is connected to.
            endNode (node): end node of the branch
            length (float): length of the branch in m
            radius (float): radius of the branch in m
            discretization (int): number of segements the dart model should have for this branch.
        """
        segmentLength = branchLength / discretization
        edgeWeihgts = {"length": segmentLength, "radius": radius}
        if startNode is None:
            nodes = []
            rootNode = node()
            nodes.append(rootNode)
            for i in range(discretization):
                newNode = node(nodes[-1], edgeWeihgts)
                nodes.append(newNode)
            self.branches.append(
                branch("Branch_0", rootNode, nodes[-1], branchLength, radius)
            )
        elif endNode is None:
            nodes = []
            nodes.append(startNode)
            for i in range(discretization):
                newNode = node(nodes[-1], edgeWeihgts)
                nodes.append(newNode)
            self.branches.append(
                branch(
                    "Branch_" + str(self.branches[-1].ID),
                    startNode,
                    nodes[-1],
                    branchLength,
                    radius,
                )
            )
        else:
            self.branches.append(
                branch(
                    "Branch_" + str(self.branches[-1].ID),
                    startNode,
                    endNode,
                    branchLength,
                    radius,
                )
            )
