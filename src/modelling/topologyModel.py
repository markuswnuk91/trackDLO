import numpy as np
from warnings import warn
from scipy.sparse import csgraph


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

    def getAdjacentNodes(self):
        adjacentNodes = []
        if self.parentNode is not None:
            adjacentNodes.append(self.parentNode)
        for node in self.childNodes:
            adjacentNodes.append(node)
        return adjacentNodes

    def getNodeInfo(self):
        return self.nodeInfo

    def setNodeInfo(self, nodeInfo):
        self.nodeInfo = nodeInfo

    def getParentEdge(self):
        return self.parentEdge

    def getChildEdges(self):
        return self.childEdges

    def getEdges(self):
        if self.parentEdge is not None:
            return [self.parentEdge] + self.childEdges
        else:
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

    def getEdge(self, i: int):
        return self.edges[i]

    def getEdges(self):
        return self.edges

    def getNumEdges(self):
        return len(self.edges)

    def getBranchInfo(self):
        return self.branchInfo

    def addBranchInfo(self, key: str, info):
        if key in self.branchInfo:
            warn("Key is already in dict. The old value is overwritten")
        self.branchInfo[key] = info

    def setBranchInfo(self, branchInfo: dict):
        self.branchInfo = branchInfo

    def appendNode(self, node):
        self.memberNodes.append(self.endNode)
        self.endNode = node
        self.nodes.append(node)

    # def _findNode(self, thisNode, searchedNode, visitedNodes):
    #     visitedNodes.append[thisNode]
    #     if thisNode == searchedNode:
    #         return visitedNodes
    #     else:
    #         adjacentNodes = thisNode.getAdjacentNodes()
    #         for adjacentNode in adjacentNodes:
    #             self.findNode(adjacentNode, searchedNode, visitedNodes)
    def _collectMemberNodes(self):
        self.memberNodes = self._breadthFirstSearch(self.startNode, self.endNode)
        self.memberNodes.pop(0)  # remove start node
        self.memberNodes.pop(-1)  # remove end node

    def _breadthFirstSearch(self, startNode: node, endNode: node):
        visitedNodes = {startNode: [startNode]}
        queue = []
        queue.append(startNode)
        while len(queue) > 0:
            node = queue.pop()
            if node == endNode:
                return visitedNodes[node]
            adjacentNodes = node.getAdjacentNodes()
            for adjacentNode in adjacentNodes:
                if adjacentNode not in visitedNodes[node]:
                    visitedNodes[adjacentNode] = visitedNodes[node] + [adjacentNode]
                    queue.append(adjacentNode)
        return None

    def _commonEdge(self, thisNode: node, oterNode: node):
        """returns the common edge betwwen thisNode and otherNode

        Args:
            thisNode (node):
            oterNode (node):

        Returns:
            edge: _description_
        """
        return list(set(thisNode.getEdges()).intersection(oterNode.getEdges()))[0]

    #     while thisNode != searchedNode:
    #         thisNode.
    # def _collectMemberNodes(self):

    #     self.startNode.getAdjacentNodes()
    #     for adjacentNode in adjacentNodes:
    #         find(endNode)
    #     self.endNode = thisNode
    #     node = self.endNode
    #     while node is not self.startNode:
    #         if self.startNode.ID > self.endNode.ID:
    #             node = self.
    #         if node is not None:
    #             node = node.getAdjacentNodes()
    #             self.memberNodes.append(node)
    #         else:
    #             raise ValueError("End node and start node seem to be not connected.")
    #     self.memberNodes.pop()
    #     self.memberNodes.reverse()

    def _collectEdges(self):
        nodes = self.getNodes()
        for i, node in enumerate(nodes):
            if node != self.endNode:
                self.edges.append(self._commonEdge(node, nodes[i + 1]))

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
        leafNodeInfo: dict = None,
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
        if leafNodeInfo is None:
            self.leafNodeInfo = {}
        else:
            self.leafNodeInfo = leafNodeInfo

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

    def getLeafNodeInfo(self):
        return self.leafNodeInfo

    def setLeafNodeInfo(self, leafNodeInfo: dict):
        self.leafNodeInfo = leafNodeInfo


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
        nodeIndex: int = None,
        branch: branch = None,
        branchIndex: int = None,
        branchNodeInfo: dict = None,
        name=None,
    ):
        if name is None:
            self.name = "branchnode_" + str(branchnode.ID)
        else:
            self.name = name
        self.ID = branchnode.ID
        branchnode.ID += 1

        self.node = node
        self.branches = [] if branch is None else [branch]
        self.nodeIndex = None if nodeIndex is None else nodeIndex
        self.branchIndices = [] if branchIndex is None else [branchIndex]
        if branchNodeInfo is None:
            self.branchNodeInfo = {}
        else:
            self.branchNodeInfo = branchNodeInfo

    def getName(self):
        return self.name

    def getNode(self):
        return self.node

    def getNodeIndex(self):
        return self.nodeIndex

    def getNumBranches(self):
        return len(self.branches)

    def getBranches(self):
        return self.branches

    def getBranch(self, num):
        return self.branches[num]

    def getBranchIndices(self):
        return self.branchIndices

    def getBranchNodeInfo(self):
        return self.branchNodeInfo

    def setBranchNodeInfo(self, branchNodeInfo: dict):
        self.branchNodeInfo = branchNodeInfo

    def getBranchIndex(self, num):
        return self.branchIndices[num]

    def appendBranch(self, branch, branchIndex=None):
        self.branches.append(branch)
        if branchIndex is not None:
            self.branchIndices.append(branchIndex)

    def getAdjacentBranchEnds(self):
        adjacentBranchEnds = []
        for branch in self.branches:
            if len(self.branches) >= 1:
                startNode = branch.getStartNode()
                endNode = branch.getEndNode()
                if startNode == self.node:
                    adjacentBranchEnds.append(endNode)
                elif endNode == self.node:
                    adjacentBranchEnds.append(startNode)
                else:
                    raise ValueError(
                        "Exptected this node to be start or end node of adjacent branch."
                    )
            else:
                pass
        return adjacentBranchEnds


class topologyModel:
    """
    A topologyModel is a collection of branches, where each branch is connected to the other branches by branchnodes. Open ends of the topology tree are called leafnodes.

    Attributes:
        nodes (node): The nodes the graph consists of, ordered accoring to the rows of the adjacencyMatrix such that nodes[0] corresponds to the first row of the adjacejcy matrix
        edges (edge): edges of the graph stored in a set
        branches (branch): The branches the topology graph consists of.
        branchNodes(list(dict)): The nodes where the branches are connected.
        leafNodes (list(dict)): The nodes at the open ends of the topology.
        leafNodeInfo (list(dict)): The additional information for the leaf nodes at the open ends of the topology.
        leafNodes (list(dict)): The nodes at the open ends of the topology.
        rootNode: first node of the grahp, with no parent
        rootBranch: branch containing the rootnode
    """

    ID = 0

    def __init__(
        self,
        adjacencyMatrix: np.ndarray = None,
        treeInfo: dict = None,
        name: str = None,
    ):
        """initialization from a adjacencyMatrix

        Args:
            adjacenyMatrix (np.array): NxN symmetric graph representation of the topology.
        """
        # check if input matrix is symmetric
        if type(adjacencyMatrix) is not np.ndarray or adjacencyMatrix.ndim != 2:
            raise ValueError("The adjacency matrix must be at a 2D numpy array.")
        if not np.all(np.abs(adjacencyMatrix - adjacencyMatrix.T) < 1e-8):
            raise ValueError(
                "Got an unsymmetric adjacency matrix. This method needs a undirected graph expressed as a symmetric adjacency matrix as input."
            )
        # check if input matrix is has zeros on diagonal
        if not np.all(adjacencyMatrix.diagonal() == 0):
            raise ValueError(
                "Got non zero entries on diagonal of adjacency matrix. This method requires a adjacency matrix with zero entries on the diagonal."
            )
        self.ID = topologyModel.ID
        topologyModel.ID += 1
        self.nodes = []
        self.edges = set()
        self.branches = []
        self.leafNodes = []
        self.branchNodes = []
        self.leafNodeInfo = []
        self.branchNodeInfo = []
        if name is None:
            self.name = "topologyModel_" + str(topologyModel.ID)
        else:
            self.name = name
        if treeInfo is None:
            self.treeInfo = {}
        else:
            self.treeInfo = treeInfo
        self.adjacencyMatrix = adjacencyMatrix

        unvisitedNodesIdxs = list(range(len(adjacencyMatrix)))
        blacklist = np.array(
            []
        )  # elements for which nodes were already generated are blacklisted
        # make sure we start with a node that is a end node
        for i in range(adjacencyMatrix.shape[0]):
            if len(np.where(adjacencyMatrix[i, :] != 0)[0]) == 1:
                break
        nextNodeCandidateIdxs = [i]

        self.nodes = [None] * len(adjacencyMatrix)
        # self.nodes[0] = rootNode
        # blacklist = np.append(blacklist, 0)

        # 1) build nodes from adjacency matrix
        while len(unvisitedNodesIdxs) > 0:
            currentNodeIdx = nextNodeCandidateIdxs.pop(0)
            if all(elem is None for elem in self.nodes):
                rootNode = node(name=str(self.name) + "_Node_0")
                self.rootNode = rootNode
                self.nodes[currentNodeIdx] = rootNode
                blacklist = np.append(blacklist, currentNodeIdx)
                currentNode = rootNode
            else:
                currentNode = self.nodes[currentNodeIdx]
            adjacentNodeIdxs = np.flatnonzero(adjacencyMatrix[currentNodeIdx, :])
            newNodeCandidateIdxs = np.setdiff1d(adjacentNodeIdxs, blacklist)
            # parentNodeIdx = np.intersect1d(adjacentNodeIdxs, blacklist)
            # self.nodes(parentNodeIdx)
            # currentNode = self.nodes[currentNodeIdx]
            # np.setdiff1d(adjacentNodeIdxs, blacklist)
            for nodeIdx in newNodeCandidateIdxs:
                newNode = node(
                    currentNode,
                    name=str(self.name) + "_Node_" + str(nodeIdx),
                    edgeInfo={"length": adjacencyMatrix[currentNodeIdx, nodeIdx]},
                )
                self.nodes[nodeIdx] = newNode
                self.edges.add(newNode.parentEdge)
                nextNodeCandidateIdxs.append(nodeIdx)
                blacklist = np.append(blacklist, nodeIdx)
            unvisitedNodesIdxs.remove(currentNodeIdx)

        # 2) find branches and identify branch and leaf nodes.
        for idx, branchCandidate in enumerate(self.nodes):
            if branchCandidate.getNumEdges() > 2:
                newBranchNode = branchnode(node=branchCandidate, nodeIndex=idx)
                self.branchNodes.append(newBranchNode)
        for thisNode in self._getBranchNodesAsNodes():
            nextBranchOrLeafNodes = self._findNextBranchOrLeafNodes(thisNode)
            for branchOrLeafNode in nextBranchOrLeafNodes:
                if not (
                    branchOrLeafNode.getNumEdges() == 1
                    or branchOrLeafNode.getNumEdges() > 2
                ):
                    raise ValueError(
                        "Got a node with {} childs but expected branch or leafnode with 0 or >2 connections.".format(
                            branchOrLeafNode.getNumEdges()
                        )
                    )

                elif self.isBranchNode(branchOrLeafNode):
                    thisBranchNode = self._getBranchNodeFromNode(thisNode)
                    otherBranchNode = self._getBranchNodeFromNode(branchOrLeafNode)
                    if thisNode in otherBranchNode.getAdjacentBranchEnds():
                        # case if branch already exists
                        pass
                    else:
                        newBranch = branch(
                            startNode=thisNode,
                            endNode=branchOrLeafNode,
                            name=self.name + "_Branch_" + str(len(self.branches)),
                        )
                        branchLength = 0
                        for edges in newBranch.getEdges():
                            branchLength += edges.getEdgeInfo()["length"]
                        newBranch.setBranchInfo({"length": branchLength})
                        self.branches.append(newBranch)
                        thisBranchNode.appendBranch(newBranch, len(self.branches) - 1)
                        otherBranchNode.appendBranch(newBranch, len(self.branches) - 1)
                else:
                    newBranch = branch(
                        startNode=thisNode,
                        endNode=branchOrLeafNode,
                        name=self.name + "_Branch_" + str(len(self.branches)),
                    )
                    # set branch length
                    branchLength = 0
                    for edges in newBranch.getEdges():
                        branchLength += edges.getEdgeInfo()["length"]
                    newBranch.setBranchInfo({"length": branchLength})
                    self.branches.append(newBranch)
                    newLeafNode = leafnode(
                        node=branchOrLeafNode,
                        nodeIndex=self.getNodeIndex(branchOrLeafNode),
                        branch=newBranch,
                        branchIndex=len(self.branches) - 1,
                    )
                    self.leafNodes.append(newLeafNode)
                    # append branch to branchNode
                    thisBranchNode = self._getBranchNodeFromNode(thisNode)
                    thisBranchNode.appendBranch(newBranch, len(self.branches) - 1)
                    # add as rootBranch if branch contains the root Node
                    if branchOrLeafNode == rootNode:
                        self.rootBranch = newBranch

        # for thisNode in self.nodes:
        #     if thisNode.getNumChilds() >= 2 or thisNode == rootNode:
        #         nextBranchOrLeafNodes = self._findNextBranchOrLeafNodes(thisNode)
        #         for branchOrLeafNode in nextBranchOrLeafNodes:
        #             if not (
        #                 branchOrLeafNode.getNumChilds() == 0
        #                 or branchOrLeafNode.getNumChilds() >= 2
        #             ):
        #                 raise ValueError(
        #                     "Got a node with {} childs but expected branch or leafnode with 0 or >2 childs.".format(
        #                         branchOrLeafNode.getNumChilds()
        #                     )
        #                 )
        #             else:
        #                 newBranch = branch(
        #                     startNode=thisNode,
        #                     endNode=branchOrLeafNode,
        #                     name=self.name + "_Branch_" + str(len(self.branches)),
        #                 )
        #                 # set branch length
        #                 branchLength = 0
        #                 for edges in newBranch.getEdges():
        #                     branchLength += edges.getEdgeInfo()["length"]
        #                 newBranch.setBranchInfo({"length": branchLength})
        #                 self.branches.append(newBranch)

        #                 # determine the node type of the beginning of the branch
        #                 if (
        #                     thisNode == rootNode
        #                     and rootNode.getNumChilds() > 2
        #                     and len(self.branchNodes) == 0
        #                 ):
        #                     #  case if rootnode is also a branch node
        #                     self.branchNodes.append(
        #                         branchnode(
        #                             node=rootNode,
        #                             nodeIndex=self.getNodeIndex(rootNode),
        #                             branch=newBranch,
        #                             branchIndex=len(self.branches) - 1,
        #                         )
        #                     )
        #                 elif (
        #                     thisNode == rootNode
        #                     and rootNode.getNumChilds() == 1
        #                     and len(self.leafNodes) == 0
        #                 ):
        #                     #  case if rootnode is a leafnode
        #                     self.leafNodes.append(
        #                         leafnode(
        #                             node=rootNode,
        #                             nodeIndex=self.getNodeIndex(rootNode),
        #                             branch=newBranch,
        #                             branchIndex=len(self.branches) - 1,
        #                         )
        #                     )
        #                 elif thisNode.getNumChilds() > 2:
        #                     thisBranchNode = self._getBranchNodeFromNode(thisNode)
        #                     thisBranchNode.appendBranch(
        #                         newBranch, len(self.branches) - 1
        #                     )
        #                 else:
        #                     pass

        #                 # determine the node type of the end of the branch
        #                 if branchOrLeafNode.getNumChilds() == 0:  # leaf node
        #                     newLeafNode = leafnode(
        #                         node=branchOrLeafNode,
        #                         nodeIndex=self.getNodeIndex(branchOrLeafNode),
        #                         branch=newBranch,
        #                         branchIndex=len(self.branches) - 1,
        #                     )
        #                     self.leafNodes.append(newLeafNode)
        #                 else:  # branch node
        #                     newBranchNode = branchnode(
        #                         node=branchOrLeafNode,
        #                         nodeIndex=self.getNodeIndex(branchOrLeafNode),
        #                         branch=newBranch,
        #                         branchIndex=len(self.branches) - 1,
        #                     )
        #                     self.branchNodes.append(newBranchNode)
        #     else:
        #         pass

    def __getitem__(self, num):
        return self.nodes[num]

    def getRootNode(self):
        return self.rootNode

    def getNumNodes(self):
        return len([node for node in self.nodes if node is not None])

    def getNode(self, num):
        return self.nodes[num]

    def getNodes(self):
        return self.nodes

    def getEdges(self):
        return self.edges

    def getNumBranches(self):
        return len(self.branches)

    def getBranches(self):
        return self.branches

    def getBranch(self, num: int):
        return self.branches[num]

    def getRootBranch(self):
        """Returns the branch that contains the root node as leaf"""
        return self.rootBranch

    def getLeafNodes(self):
        return self.leafNodes

    def getBranchNodes(self):
        return self.branchNodes

    def getNumBranchNodes(self):
        return len(self.branchNodes)

    def getNumLeafNodes(self):
        return len(self.leafNodes)

    def _getBranchNodeFromNode(self, node: node):
        """
        Internal Method, which is not recommended for external use.
        Returns the BranchNode corresponding to a given node from the
        list of branchNodes.
        Args:
            node (node): node for which the branchNode should be determined

        Returns:
            branchNode (branchNode): branchNode
        """
        branchNodeList = self._getBranchNodesAsNodes()
        branchNodeIdx = branchNodeList.index(node)
        return self.branchNodes[branchNodeIdx]

    def _getBranchNodesAsNodes(self):
        branchNodesList = []
        for branchNode in self.branchNodes:
            branchNodesList.append(branchNode.getNode())
        return branchNodesList

    def _getLeafNodesAsNodes(self):
        leafNodeList = []
        for leafNode in self.leafNodes:
            leafNodeList.append(leafNode.getNode())
        return leafNodeList

    def _findNextBranchOrLeafNodes(self, thisNode, previousNode=None):
        nextBranchOrLeafNodes = []
        if (
            thisNode.getNumEdges() == 1 or thisNode.getNumEdges() > 2
        ) and previousNode is not None:
            nextBranchOrLeafNodes.append(thisNode)
            return nextBranchOrLeafNodes
        else:
            if previousNode is None:
                adjacentNodes = thisNode.getAdjacentNodes()
            else:
                adjacentNodes = thisNode.getAdjacentNodes()
                adjacentNodes.remove(previousNode)
            for adjacentNode in adjacentNodes:
                nextBranchOrLeafNodes.append(
                    self._findNextBranchOrLeafNodes(adjacentNode, thisNode)[0]
                )

        return nextBranchOrLeafNodes

        #     adjacentNodes = thisNode.getAdjacentNodes()
        # else:
        #     if thisNode.getNumEdges() == 1 or thisNode.getNumEdges() > 2:
        #         nextBranchOrLeafNodes.append(thisNode)
        #     else:

        # for
        #         nextBranchOrLeafNodes.append(
        #             self._findNextBranchOrLeafNodes(adjacentNode, thisNode)[0]
        #         )

        # if previousNode is None:
        #     previousNode = thisNode
        # nextBranchOrLeafNodes = []
        # if thisNode.getNumEdges() == 0:
        #     return nextBranchOrLeafNodes
        # else:
        #     for adjacentNode in thisNode.getAdjacentNodes().remove(previousNode):
        #         if adjacentNode is not previousNode and (
        #             adjacentNode.getNumEdges() == 1 or adjacentNode.getNumEdges() > 2
        #         ):
        #             nextBranchOrLeafNodes.append(adjacentNode)
        #         else:  # member node
        #             nextBranchOrLeafNodes.append(
        #                 self._findNextBranchOrLeafNodes(adjacentNode, thisNode)[0]
        #             )
        #     return nextBranchOrLeafNodes

    def getLeafNodeFromNode(self, node: node):
        if node in self._getLeafNodesAsNodes():
            return self.leafNodes[self._getLeafNodesAsNodes().index(node)]
        else:
            return None

    def getBranchNodeFromNode(self, node: node):
        if node in self._getBranchNodesAsNodes():
            return self.branchNodes[self._getBranchNodesAsNodes().index(node)]
        else:
            return None

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

    def getBranchesFromNode(self, node: node):
        branchList = []
        for branch in self.branches:
            if node in branch.getNodes():
                branchList.append(branch)
        return branchList

    def isLeafNode(self, node: node):
        if node in self._getLeafNodesAsNodes():
            return True
        else:
            return False

    def isBranchNode(self, node: node):
        if node in self._getBranchNodesAsNodes():
            return True
        else:
            return False

    def getChildBranches(self, branch: branch):
        childBranches = []
        if self.isBranchNode(branch.getEndNode()):
            branchNode = self._getBranchNodeFromNode(branch.getEndNode())
            childBranches = branchNode.getBranches().copy()
            childBranches.remove(branch)
        return childBranches

    def isRootBranch(self, branch: branch):
        if (
            branch.getStartNode() == self.rootNode
            or branch.getEndNode() == self.rootNode
        ):
            return True
        else:
            return False

    def getSummedLength(self):
        length = 0
        for branch in self.branches:
            length += branch.getBranchInfo()["length"]
        return length

    def getBranchIndex(self, branch: branch):
        return self.branches.index(branch)

    def getBranchIndices(self, branches: list):
        branchIndices = []
        for branch in branches:
            branchIndices.append(self.getBranchIndex(branch))
        return branchIndices

    def getSiblingBranches(self, branch: branch):
        """
        Returns the siblings to this branch.
        Sibling branches are branches which share a start or end node with the given branch.
        """
        siblingBranches = []
        siblingCandidates = self.branches.copy()
        siblingCandidates.remove(branch)
        branchEnds = [branch.getStartNode(), branch.getEndNode()]
        for siblingCandidate in siblingCandidates:
            siblingCandidateEnds = [
                siblingCandidate.getStartNode(),
                siblingCandidate.getEndNode(),
            ]
            if any(branchEnd in branchEnds for branchEnd in siblingCandidateEnds):
                siblingBranches.append(siblingCandidate)
        return siblingBranches

    def getBranchLength(self, branchNumber: int):
        return self.getBranch(branchNumber).getBranchInfo()["length"]

    def getLocalCoordinateFromBranchNode(self, branch: branch, node: node):
        if node not in branch.getNodes():
            raise ValueError("The given node is not in the given branch")
        else:
            if node == branch.getStartNode():
                return 0
            elif node == branch.getEndNode():
                return 0
            else:
                branchLength = branch.getBranchInfo()["length"]
                localLength = 0
                nodeIdx = branch.getMemberNodes().index(node)
                for i in range(0, nodeIdx + 1):
                    localLength += branch.getEdge(i).getEdgeInfo()["length"]
            return localLength / branchLength
