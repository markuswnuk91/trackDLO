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
        if parentNode is not None and (not isinstance(parentNode, node)):
            raise ValueError(
                "Expected the parentNode to be of type node. Instead got {}".format(
                    type(parentNode)
                )
            )

        if name is None:
            self.name = "Node_" + str(node.ID)
        else:
            self.name = name
        self.ID = node.ID
        node.ID += 1

        self.parentNode = parentNode

        self.childNodes = []
        self.childEdges = []
        if parentNode is not None:
            self.parentEdge = edge(self.parentNode, self, edgeInfo)
            self.parentNode._addChildNode(self)
        else:
            self.parentEdge = None
            if edgeInfo is not None:
                warn(
                    "Edge info was given but no parent node was specified. Edge info is discarded."
                )
        if nodeInfo is None:
            self.nodeInfo = {}
        else:
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

    def getNumChilds(self):
        return len(self.childNodes)

    def getNumEdges(self):
        if self.parentNode is None:
            return len(self.childNodes)
        else:
            return len(self.childNodes) + 1

    def getEdgeInfo(self, num: int):
        """
        get the info of corresponding edges
        0 refers to the parent edge. 1 refers to the first child edge
        """
        if self.parentNode is None and num == 0:
            return {}
        elif self.parentNode is not None and num == 0:
            return self.getParentEdge().getEdgeInfo()
        elif len(self.getChildEdges()) > 0:
            return self.getChildEdges()[num - 1].getEdgeInfo()
        else:
            warn(
                "Node has only {} edges, but edge {} was requested".format(
                    self.getNumEdges(), num
                )
            )
            return None

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

        if name is None:
            self.name = "Edge_" + str(parentNode.ID) + "_to_" + str(childNode.ID)
        else:
            self.name = name
        self.ID = edge.ID
        edge.ID += 1

        self.parentNode = parentNode
        self.childNode = childNode
        self.edgeInfo = edgeInfo
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
        edges (list of edges): The edges the branch consists of.
    """

    ID = 0

    def __init__(
        self,
        numNodes: int = None,
        startNode: node = None,
        endNode: node = None,
        branchInfo: dict = None,
        name: str = None,
    ):

        if name is None:
            self.name = "Branch_" + str(branch.ID)
        else:
            self.name = name

        self.ID = branch.ID
        branch.ID += 1

        if branchInfo is None:
            self.branchInfo = {}
        else:
            self.branchInfo = branchInfo
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
                newNode = node(self.startNode)
                for i in range(numNodes - 2):
                    i += 1
                    self.memberNodes.append(newNode)
                    newNode = node(newNode)
                self.endNode = newNode
        elif endNode is None and numNodes is None:
            self.endNode = node(self.startNode)
        elif endNode is not None and numNodes is not None:
            self.endNode = endNode
            self._collectMemberNodes()
            if len(self.memberNodes) + 2 != numNodes:
                warn(
                    "Number of nodes given does not equal the desired number of nodes of the branch. Expected number of nodes is: {}, number of Nodes from startNode to endNode are {}.".format(
                        numNodes, len(self.memberNodes) + 2
                    )
                )
        else:
            self.endNode = endNode
            self._collectMemberNodes()

        self.nodes = self.memberNodes.copy()
        self.nodes.insert(0, startNode)
        self.nodes.append(endNode)
        self._collectEdges()

    def getName(self):
        return self.name

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

    def getEdges(self):
        return self.edges

    def getNumEdges(self):
        return len(self.edges)

    def getBranchInfo(self):
        return self.branchInfo

    def setBranchInfo(self, branchInfo: dict):
        self.branchInfo = branchInfo

    def appendNode(self, node):
        self.memberNodes.append(self.endNode)
        self.endNode = node
        self.nodes.append(node)

    def _collectMemberNodes(self):
        node = self.endNode
        while node is not self.startNode:
            if node is not None:
                node = node.getParent()
                self.memberNodes.append(node)
            else:
                raise ValueError("End node and start node seem to be not connected.")
        self.memberNodes.pop()
        self.memberNodes.reverse()

    def _collectEdges(self):
        node = self.endNode
        while node is not self.startNode:
            if node is not None:
                self.edges.append(node.getParentEdge())
                node = node.getParent()
        self.edges.reverse()

    def __str__(self):
        _str = ""
        _str += "Branch ID: {}".format(self.ID)
        return _str


class leafnode:
    """
    LeafNodes are nodes at the ends of a topology graph.
    The leafnode is associated with a node and a branch.
    It stores information to which node of the graph it refers to and to which branch it belongs.

    Attributes:
        node (node): The node the leaf belongs to
        nodeIndex (int): The index to access the leafnode in the graph
        branch (branch): The branch this leafnode belongs to.
        branchIndex (int): The index to access the branch in the graph
        leafnodeInfo (dict): Optional additional information to store in a leafnode.
    """

    ID = 0

    def __init__(
        self,
        node: node,
        nodeIndex: int,
        branch: branch,
        branchIndex: int,
        leafnodeInfo: dict = None,
        name=None,
    ):
        if name is None:
            self.name = "leafnode_" + str(leafnode.ID)
        else:
            self.name = name
        self.ID = leafnode.ID
        leafnode.ID += 1

        self.node = node
        self.branch = branch
        self.nodeIndex = nodeIndex
        self.branchIndex = branchIndex
        if leafnodeInfo is None:
            self.leafnodeInfo = {}
        else:
            self.leafnodeInfo = leafnodeInfo

    def getName(self):
        return self.name

    def getNode(self):
        return self.node

    def getNodeIndex(self):
        return self.nodeIndex

    def getBranch(self):
        return self.branch

    def getBranchIndex(self):
        return self.branchIndex


class branchnode:
    """
    BranchNodes are nodes at the the branch poitns of a topology graph.
    The branch node is associated with a node and several branches.
    It stores information to which node of the graph it refers to and to which branches it belongs.

    Attributes:
        node (node): The node the branchnode belongs to
        nodeIndex (int): The index to access the branchnode in the graph
        branches (list(branch)): The branches this branchnode belongs to.
        branchIndices (list(int)): The indices to access the branches in the graph
        branchnodeInfo (dict): Optional additional information to store in a bracnhnode.
    """

    ID = 0

    def __init__(
        self,
        node: node,
        nodeIndex: int,
        branch: branch,
        branchIndex: int,
        branchnodeInfo: dict = None,
        name=None,
    ):
        if name is None:
            self.name = "branchnode_" + str(branchnode.ID)
        else:
            self.name = name
        self.ID = branchnode.ID
        branchnode.ID += 1

        self.node = node
        self.branches = [branch]
        self.nodeIndex = nodeIndex
        self.branchIndices = [branchIndex]
        if branchnodeInfo is None:
            self.branchnodeInfo = {}
        else:
            self.branchnodeInfo = branchnodeInfo

    def getName(self):
        return self.name

    def getNode(self):
        return self.node

    def getNodeIndex(self):
        return self.nodeIndex

    def getBranches(self):
        return self.branches

    def getBranch(self, num):
        return self.branches[num]

    def getBranchIndices(self):
        return self.branchIndices

    def getBranchIndex(self, num):
        return self.branchIndices[num]

    def appendBranch(self, branch, branchIndex):
        self.branches.append(branch)
        self.branchIndices.append(branchIndex)


class topologyTree:
    """
    A topologyTree is a collection of branches, where each branch is connected to the other branches by branchnodes. Open ends of the topology tree are called leafnodes.

    Attributes:
        nodes (node): The nodes the graph consists of.
        branches (branch): The branches the topology graph consists of.
        branchNodes(list(dict)): The nodes where the branches are connected.
        leafNodes (list(dict)): The nodes the at the open ends of the topology.
        leafNodeInfo (list(dict)): The additional information for the leaf nodes the at the open ends of the topology.
        leafNodes (list(dict)): The nodes the at the open ends of the topology.
    """

    ID = 0

    def __init__(self, csgraph: np.array = None, name: str = None):
        """initialization from a adjacencyMatrix

        Args:
            adjacenyMatrix (np.array): NxN symmetric graph representation of the topology.
        """
        self.ID = topologyTree.ID
        topologyTree.ID += 1
        self.nodes = []
        self.branches = []
        self.leafNodes = []
        self.branchNodes = []
        self.leafNodeInfo = []
        self.branchNodeInfo = []

        if name is None:
            self.name = "TopologyTree_" + str(topologyTree.ID)
        else:
            self.name = name

        # 1) build nodes from adjacency matrix
        if not np.all(
            np.abs(csgraph - csgraph.T) < 1e-8
        ):  # check if input matrix is symmetric
            raise ValueError(
                "Got an unsymmetric adjacency matrix. This method needs a symmetric cs graph as input."
            )
        unvisitedNodesIdxs = list(range(len(csgraph)))
        blacklist = np.array(
            []
        )  # elements for which were already generated are blacklisted
        self.nodes = [None] * len(csgraph)
        rootNode = node(name=str(self.name) + "_Node_0")
        currentNode = rootNode
        nextNodeCandidateIdxs = [0]
        self.nodes[0] = rootNode
        blacklist = np.append(blacklist, 0)

        while len(unvisitedNodesIdxs) > 0:
            currentNodeIdx = nextNodeCandidateIdxs.pop(0)
            currentNode = self.nodes[currentNodeIdx]
            if currentNode is None:
                raise ValueError(
                    "Got an empty node. Make sure given csgraph is correctly connected."
                )
            adjacentNodeIdxs = np.flatnonzero(csgraph[currentNodeIdx, :])
            newNodeCandidateIdxs = np.setdiff1d(adjacentNodeIdxs, blacklist)
            for nodeIdx in newNodeCandidateIdxs:
                newNode = node(
                    currentNode,
                    name=str(self.name) + "_Node_" + str(nodeIdx),
                    edgeInfo={"length": csgraph[currentNodeIdx, nodeIdx]},
                )
                self.nodes[nodeIdx] = newNode
                nextNodeCandidateIdxs.append(nodeIdx)
                blacklist = np.append(blacklist, nodeIdx)
            unvisitedNodesIdxs.remove(currentNodeIdx)

        # 2) find branches and identify branch and leaf nodes.
        for thisNode in self.nodes:
            if thisNode.getNumChilds() >= 2 or thisNode == rootNode:
                nextBranchOrLeafNodes = self._findNextBranchOrLeafNodes(thisNode)
                for branchOrLeafNode in nextBranchOrLeafNodes:
                    if not (
                        branchOrLeafNode.getNumChilds() == 0
                        or branchOrLeafNode.getNumChilds() >= 2
                    ):
                        raise ValueError(
                            "Got a node with {} childs but expected branch or leafnode with 0 or >2 childs.".format(
                                branchOrLeafNode.getNumChilds()
                            )
                        )
                    else:
                        newBranch = branch(
                            startNode=thisNode,
                            endNode=branchOrLeafNode,
                            name=self.name + "_Branch_" + str(len(self.branches)),
                        )
                        branchLength = 0
                        for edges in newBranch.getEdges():
                            branchLength += edges.getEdgeInfo()["length"]
                        newBranch.setBranchInfo({"branchLength": branchLength})
                        self.branches.append(newBranch)

                        if (
                            thisNode == rootNode
                            and rootNode.getNumChilds() >= 2
                            and len(self.branchNodes) == 0
                        ):
                            #  case if rootnode is also a branch node
                            self.branchNodes.append(
                                branchnode(
                                    node=rootNode,
                                    nodeIndex=self.getNodeIndex(rootNode),
                                    branch=newBranch,
                                    branchIndex=len(self.branches) - 1,
                                )
                            )
                        elif (
                            thisNode == rootNode
                            and rootNode.getNumChilds() == 1
                            and len(self.leafNodes) == 0
                        ):
                            #  case if rootnode is a leafnode
                            self.leafNodes.append(
                                leafnode(
                                    node=rootNode,
                                    nodeIndex=self.getNodeIndex(rootNode),
                                    branch=newBranch,
                                    branchIndex=len(self.branches) - 1,
                                )
                            )
                        elif thisNode.getNumChilds() >= 2:
                            thisBranchNode = self._getBranchNodeFromNode(thisNode)
                            thisBranchNode.appendBranch(
                                newBranch, len(self.branches) - 1
                            )
                        else:
                            pass

                        if branchOrLeafNode.getNumChilds() == 0:  # leaf node
                            newLeafNode = leafnode(
                                node=branchOrLeafNode,
                                nodeIndex=self.getNodeIndex(branchOrLeafNode),
                                branch=newBranch,
                                branchIndex=len(self.branches) - 1,
                            )
                            self.leafNodes.append(newLeafNode)
                        else:  # branch node
                            # check if new branch node required or if it already exists
                            newBranchNode = branchnode(
                                node=branchOrLeafNode,
                                nodeIndex=self.getNodeIndex(branchOrLeafNode),
                                branch=newBranch,
                                branchIndex=len(self.branches) - 1,
                            )
                            self.branchNodes.append(newBranchNode)
                            thisNode
            else:
                pass

    def __getitem__(self, num):
        return self.nodes[num]

    def getNumNodes(self):
        return len([node for node in self.nodes if node is not None])

    def getNode(self, num):
        return self.nodes[num]

    def getNodes(self):
        return self.nodes

    def getNumBranches(self):
        return len(self.branches)

    def getBranches(self):
        return self.branches

    def getBranch(self, num: int):
        return self.branches[num]

    def getLeafNodes(self):
        return self.leafNodes

    def getBranchNodes(self):
        return self.branchNodes

    def getNumBranchNodes(self):
        return len(self.branchNodes)

    def getNumLeafNodes(self):
        return len(self.leafNodes)

    def _getBranchNodeFromNode(self, node: node):
        branchNodeList = self._getBranchNodesAsNodes()
        branchNodeIdx = branchNodeList.index(node)
        return self.branchNodes[branchNodeIdx]

    def _getBranchNodesAsNodes(self):
        branchNodesList = []
        for branchNode in self.branchNodes:
            branchNodesList.append(branchNode.getNode())
        return branchNodesList

    def getLeafNodeIndices(self):
        leafNodeIndices = []
        for leafNode in self.leafNodes:
            leafNodeIndex = leafNode.getNodeIndex()
            leafNodeIndices.append(leafNodeIndex)
        return leafNodeIndices

    def getBranchNodeIndices(self):
        branchNodeIndices = []
        for branchNode in self.branchNodes:
            branchNodeIndex = branchNode.getNodeIndex()
            branchNodeIndices.append(branchNodeIndex)
        return branchNodeIndices

    def getNodeIndex(self, node: node):
        return self.nodes.index(node)

    def _findNextBranchOrLeafNodes(self, thisNode):
        nextBrachOrLeafNodes = []
        if thisNode.getNumChilds() < 1:
            return nextBrachOrLeafNodes
        else:
            for childNode in thisNode.getChilds():
                if childNode.getNumChilds() == 0 or childNode.getNumChilds() >= 2:
                    nextBrachOrLeafNodes.append(childNode)
                else:  # member node
                    nextBrachOrLeafNodes.append(
                        self._findNextBranchOrLeafNodes(childNode)[0]
                    )
            return nextBrachOrLeafNodes


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
