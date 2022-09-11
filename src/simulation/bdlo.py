import dartpy as dart
import numpy as np
from warnings import warn


class node:
    """
    Nodes are the basic elements of a graph.
    Each node contains the information to which parent it is connected.

    Attributes:
        parentNode (node): The parent of this node
        edges (edge): The edges connecting this node with its parent node and child nodes. parent edge is the fist edge in self.
        edgeInfo (dict): information to be stored in the edge connectiong this node to the parent node
        nodeInfo (dict): information to be stored in this node

    """

    ID = 0

    def __init__(
        self, parentNode=None, edgeInfo: dict = None, nodeInfo: dict = None, name=None
    ):
        self.ID = node.ID
        node.ID += 1
        if name is None:
            self.name = "Node_" + str(self.ID)
        else:
            self.name = name

        self.parentNode = parentNode

        self.childNodes = []
        self.childEdges = []
        if parentNode is not None:
            self.parentEdge = edge(self.parentNode, self, edgeInfo)
            self.parentNode._addChildNode(self)
        else:
            self.parentEdge = None

        self.nodeInfo = nodeInfo

    def getName(self):
        return self.name

    def getParent(self):
        return self.parentNode

    def getChilds(self):
        return self.childNodes

    def getNodeInfo(self):
        return self.nodeInfo

    def setNodeInfo(self, nodeInfo):
        self.nodeInfo = nodeInfo

    def getParentEdge(self):
        return self.parentEdge

    def getChildEdges(self):
        return self.childEdges

    def hasParentNode(self):
        return self.parentNode is not None

    def _addChildNode(self, node):
        self.childNodes.append(node)
        self.childEdges.append(node.parentEdge)


class edge:
    """
    Edges are the basic elements of a graph.
    Each edge connects two nodes.
    It holds additional information such as weigts.

    Attributes:
        parentNode (node): The parent node of this edge
        childNode (node): The child node of this edge.
        edgeInfo (dict): Information to be stored in this edge.
    """

    ID = 0

    def __init__(
        self, parentNode: node, childNode: node, edgeInfo: dict = None, name: str = None
    ):
        self.ID = edge.ID
        edge.ID += 1
        if name is None:
            self.name = "Edge_" + str(parentNode.ID) + "_to_" + str(childNode.ID)
        else:
            self.name = name
        self.parentNode = parentNode
        self.childNode = childNode
        self.edgeInfos = edgeInfo
        if edgeInfo is not None:
            self.edgeInfo = edgeInfo

    def getName(self):
        return self.name

    def getParentNode(self):
        return self.parentNode

    def getChildNode(self):
        return self.childNode

    def getEdgeInfo(self):
        return self.edgeInfo

    def setEdgeInfo(self, edgeInfo):
        self.edgeInfo = edgeInfo


class branch:
    """
    A branch is a collection of nodes and edges, where each node is connected to its adjacent node by exactly one edge.

    Attributes:
        numNodes: The number of nodes the branch should consist of.
        startNode (node): The node the branch starts at.
        endNode (node): The node the branch ends at.
        memberNodes (list of nodes): The member nodes of thse branch (exclusively the start and end node).
    """

    ID = 0

    def __init__(
        self,
        numNodes: int = None,
        startNode: node = None,
        endNode: node = None,
        name: str = None,
    ):
        self.ID = branch.ID
        branch.ID += 1

        self.memberNodes = []
        self.nodes = []
        self.edges = []

        if startNode is None:
            self.startNode = node()
        else:
            self.startNode = startNode

        if endNode is None and numNodes is not None:
            if numNodes <= 1:
                raise ValueError(
                    "Expected an integer larger than 1 for numNodes instead got: {}".format(
                        numNodes
                    )
                )
            else:
                newNode = node(startNode)
                for i in range(numNodes - 2):
                    self.memberNodes.append(newNode)
                    newNode = node(self.newNode)
                self.endNode = newNode
        elif endNode is None and numNodes is None:
            self.endNode = node(self.startNode)
        elif endNode is not None and numNodes is not None:
            self.endNode = endNode
            self.memberNodes = self._collectMemberNodes()
            if len(self.memberNodes) + 2 != numNodes:
                warn(
                    "Number of nodes given does not equal the desired number of nodes of the branch. Expected number of nodes is: {}, number of Nodes from startNode to endNode are {}.".format(
                        numNodes, len(self.memberNodes) + 2
                    )
                )
        else:
            self.endNode = endNode
            self.memberNodes = self._collectMemberNodes()

        if name is None:
            self.name = (
                "Branch_"
                + "Node_"
                + str(startNode.ID)
                + "_to_"
                + "Node_"
                + str(endNode.ID)
            )
        else:
            self.name = name

        self.nodes = self.memberNodes
        self.nodes.insert(0, startNode)
        self.nodes.append(endNode)
        self._collectEdges()

    def getStartNode(self):
        return self.startNode

    def getEndNode(self):
        return self.endNode

    def getMemberNodes(self):
        return self.memberNodes

    def getNodes(self):
        return self.nodes

    def getNumNodes(self):
        return len(self.nodes)

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
        node = self.endNode
        while node is not self.startNode:
            if node is not None:
                self.edges.append(node.getParentEdge())
                node = node.getParentNode()
        self.edges.reverse()

    def __str__(self):
        _str = ""
        _str += "Branch ID: {}".format(self.ID)
        return _str


class tree:

    self.branches = []
    self.leafNodes = []
    self.branchNodes = []


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
