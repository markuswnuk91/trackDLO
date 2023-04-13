import numpy as np
from warnings import warn
from scipy.sparse import csgraph


class Node:
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

    def __init__(self, otherNode=None, nodeInfo: dict = None, name=None):
        if otherNode is not None and (not isinstance(otherNode, Node)):
            raise ValueError(
                "Expected the parentNode to be of type node. Instead got {}".format(
                    type(otherNode)
                )
            )

        if name is None:
            self.name = "Node_" + str(Node.ID)
        else:
            self.name = name
        self.ID = Node.ID
        Node.ID += 1
        self.edges = []

        if otherNode is not None:
            newEdge = Edge([otherNode, self])

        if nodeInfo is None:
            self.nodeInfo = {}
        else:
            self.nodeInfo = nodeInfo

    def getName(self):
        return self.name

    def getEdges(self):
        return self.edges

    def getNumEdges(self):
        return len(self.edges)

    def addEdge(self, edge):
        self.edges.append(edge)

    def getNodeInfo(self):
        return self.nodeInfo

    def setNodeInfo(self, nodeInfo):
        self.nodeInfo = nodeInfo

    def addNodeInfo(self, key: str, info):
        if key in self.nodeInfo:
            warn("Key is already in dict. The old value is overwritten")
        self.nodeInfo[key] = info

    def getAdjacentNode(self, edge):
        edgeNodes = edge.getNodes()
        thisNodeIdx = edgeNodes.index(self)
        if thisNodeIdx == 0:
            return edgeNodes[1]
        else:
            return edgeNodes[0]

    def getAdjacentNodes(self):
        adjacentNodes = []
        for edge in self.edges:
            edgeNodes = edge.getNodes()
            for node in edgeNodes:
                if node is not self:
                    adjacentNodes.append(node)
        return adjacentNodes

    def hasEdgeWith(self, otherNode):
        hasEdge = False
        for edge in self.edges:
            if otherNode == self.getAdjacentNode(edge):
                hasEdge = True
        return hasEdge


class Edge:
    """
    Edges are the basic elements of a graph.
    Each edge connects two nodes.
    It holds additional information such as weigts.

    Attributes:
        connectedNodes (list(node)): List of nodes the edge connects
        edgeInfo (dict): Information to be stored in this edge.
    """

    ID = 0

    def __init__(self, nodes: tuple, edgeInfo: dict = None, name: str = None):
        if len(nodes) != 2:
            raise ValueError(
                "Edges connect two nodes. Indest got {} nodes.".format(len(nodes))
            )
        self.ID = Edge.ID
        Edge.ID += 1
        if name is None:
            self.name = "Edge_Node" + str(nodes[0].ID) + "_Node" + str(nodes[1].ID)
        else:
            self.name = name
        self.nodes = nodes
        self.edgeInfo = edgeInfo
        if edgeInfo is not None:
            self.edgeInfo = edgeInfo

        nodes[0].addEdge(self)
        nodes[1].addEdge(self)

    def getName(self):
        return self.name

    def getNodes(self):
        return self.nodes

    def getEdgeInfo(self):
        return self.edgeInfo

    def setEdgeInfo(self, edgeInfo):
        self.edgeInfo = edgeInfo


class branch:
    """
    A branch is a collection of nodes and edges, where each node is connected to its adjacent node by exactly one edge.

    Attributes:
        numNodes: The number of nodes the branch should consist of.
        startNode (node): The node the branch starts at. In general this is a branch node. The only exception is the rootBranch which starts with the rootNode as a leafNode.
        endNode (node): The node the branch ends at.
        memberNodes (list of nodes): The member nodes of thse branch (exclusively the start and end node).
        edges (list of edges): The edges the branch consists of.
    """

    ID = 0

    def __init__(
        self,
        numNodes: int = None,
        startNode: Node = None,
        endNode: Node = None,
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
            self.startNode = Node()
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
                newNode = Node(self.startNode)
                for i in range(numNodes - 2):
                    i += 1
                    self.memberNodes.append(newNode)
                    newNode = Node(newNode)
                self.endNode = newNode
        elif endNode is None and numNodes is None:
            self.endNode = Node(self.startNode)
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

    def _collectMemberNodes(self):
        self.memberNodes = self._breadthFirstSearch(self.startNode, self.endNode)
        self.memberNodes.pop(0)  # remove start node
        self.memberNodes.pop(-1)  # remove end node

    def _breadthFirstSearch(self, startNode: Node, endNode: Node):
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

    def _commonEdge(self, thisNode: Node, oterNode: Node):
        """returns the common edge betwwen thisNode and otherNode

        Args:
            thisNode (node):
            oterNode (node):

        Returns:
            edge: edge connecting the two nodes
        """
        return list(set(thisNode.getEdges()).intersection(oterNode.getEdges()))[0]

    def _collectEdges(self):
        nodes = self.getNodes()
        for i, node in enumerate(nodes):
            if node != self.endNode:
                self.edges.append(self._commonEdge(node, nodes[i + 1]))

    def __str__(self):
        _str = ""
        _str += "Branch ID: {}".format(self.ID)
        return _str

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
        node: Node,
        nodeIndex: int = None,
        branch: branch = None,
        branchIndex: int = None,
        leafNodeInfo: dict = None,
        name=None,
    ):
        self.ID = leafnode.ID
        leafnode.ID += 1

        self.node = node
        self.branch = None if branch is None else branch
        self.nodeIndex = None if nodeIndex is None else nodeIndex
        self.branchIndex = None if branchIndex is None else branchIndex

        if name is None:
            self.name = "leafnode_" + str(leafnode.ID)
        else:
            self.name = name

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
        return

    def appendBranch(self, branch, branchIndex=None):
        if self.branch is None:
            self.branch = branch
            self.branchIndex = branchIndex
        else:
            warn("LeafNode had a branch already. Got overridden with new branch")
        return


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
        node: Node,
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


class topologyModel(object):
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
        *args,
        **kwargs,
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
        self.edges = []
        self.branches = []
        self.leafNodes = []
        self.branchNodes = []
        self.leafNodeInfo = []
        self.branchNodeInfo = []
        self.rootNode = None
        self.rootBranch = None

        if name is None:
            self.name = "topologyModel_" + str(topologyModel.ID)
        else:
            self.name = name
        if treeInfo is None:
            self.treeInfo = {}
        else:
            self.treeInfo = treeInfo
        self.adjacencyMatrix = adjacencyMatrix

        # 1) build nodes from adjacency matrix
        # generate nodes
        for i in range(0, adjacencyMatrix.shape[0]):
            newNode = Node(name=str(self.name) + "_Node_" + str(i))
            self.nodes.append(newNode)
        # generate edges
        for i, node in enumerate(self.nodes):
            adjacentNodeIdxs = np.flatnonzero(adjacencyMatrix[i, :])
            if self.rootNode is None and len(adjacentNodeIdxs) == 1:
                self.rootNode = node
            for adjacentNodeIdx in adjacentNodeIdxs:
                if node.hasEdgeWith(self.nodes[adjacentNodeIdx]):
                    pass
                else:
                    newEdge = Edge(
                        nodes=(node, self.nodes[adjacentNodeIdx]),
                        edgeInfo={"length": adjacencyMatrix[i, adjacentNodeIdx]},
                    )
                    self.edges.append(newEdge)

        # 2) identify branch and leaf nodes.
        for idx, branchNodeCandidate in enumerate(self.nodes):
            if branchNodeCandidate.getNumEdges() > 2:
                newBranchNode = branchnode(node=branchNodeCandidate, nodeIndex=idx)
                self.branchNodes.append(newBranchNode)
        for idx, leafNodeCandidate in enumerate(self.nodes):
            if leafNodeCandidate.getNumEdges() == 1:
                newLeafNode = leafnode(node=leafNodeCandidate, nodeIndex=idx)
                self.leafNodes.append(newLeafNode)

        # 3) reconstuct branches (breadth-first)
        # add the root branch first
        nextBranchOrLeafNode = self._findNextBranchOrLeafNodes(self.rootNode)
        rootBranch = self._addBranch(
            startNode=self.rootNode,
            endNode=nextBranchOrLeafNode[0],
            name=self.name + "_Branch_" + str(len(self.branches)),
        )
        self.rootBranch = rootBranch
        # go throgh the topolgoy and generate the branches for all branch points
        nextBranchNodes = [rootBranch.endNode]
        visitedNodes = [self.rootNode]
        while len(nextBranchNodes) > 0:
            thisNode = nextBranchNodes.pop(0)
            nextBranchOrLeafNodes = self.findNextBranchORLeafNodeBFS(thisNode)
            for branchOrLeafNode in nextBranchOrLeafNodes:
                if branchOrLeafNode in visitedNodes:
                    pass
                elif not (
                    branchOrLeafNode.getNumEdges() == 1
                    or branchOrLeafNode.getNumEdges() > 2
                ):
                    raise ValueError(
                        "Got a node with {} childs but expected branch or leafnode with 0 or >2 connections.".format(
                            branchOrLeafNode.getNumEdges()
                        )
                    )
                elif self.isBranchNode(branchOrLeafNode):  # branchNode
                    nextBranchNodes.append(branchOrLeafNode)
                    self._getBranchNodeFromNode(thisNode)
                    otherBranchNode = self._getBranchNodeFromNode(branchOrLeafNode)
                    if thisNode in otherBranchNode.getAdjacentBranchEnds():
                        # case if branch already exists
                        pass
                    else:
                        newBranch = self._addBranch(
                            startNode=thisNode,
                            endNode=branchOrLeafNode,
                            name=self.name + "_Branch_" + str(len(self.branches)),
                        )
                else:  # leafNode
                    self._addBranch(
                        startNode=thisNode,
                        endNode=branchOrLeafNode,
                        name=self.name + "_Branch_" + str(len(self.branches)),
                    )

            visitedNodes.append(thisNode)

    def __getitem__(self, num):
        return self.nodes[num]

    def _addBranch(self, startNode, endNode, name):
        newBranch = branch(
            startNode=startNode,
            endNode=endNode,
            name=self.name + "_Branch_" + str(len(self.branches)),
        )
        # set branch length
        branchLength = 0
        for edges in newBranch.getEdges():
            branchLength += edges.getEdgeInfo()["length"]
        newBranch.setBranchInfo({"length": branchLength})
        self.branches.append(newBranch)

        if self.isLeafNode(startNode):
            thisLeafNode = self.getLeafNodeFromNode(startNode)
            thisLeafNode.appendBranch(newBranch, len(self.branches) - 1)
        else:
            thisBranchNode = self.getBranchNodeFromNode(startNode)
            thisBranchNode.appendBranch(newBranch, len(self.branches) - 1)

        if self.isLeafNode(endNode):
            thisLeafNode = self.getLeafNodeFromNode(endNode)
            thisLeafNode.appendBranch(newBranch, len(self.branches) - 1)
        else:
            thisBranchNode = self.getBranchNodeFromNode(endNode)
            thisBranchNode.appendBranch(newBranch, len(self.branches) - 1)
        return newBranch

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

    def getNumBranchNodesFromBranch(self, branch):
        numBranchNodes = 0
        startNode = branch.getStartNode()
        endNode = branch.getEndNode()
        if self.isBranchNode(startNode):
            numBranchNodes += 1

        if self.isBranchNode(endNode):
            numBranchNodes += 1

        return numBranchNodes

    def getNumLeafNodes(self):
        return len(self.leafNodes)

    def getNumLeafNodesFromBranch(self, branch):
        numLeafNodes = 0
        startNode = branch.getStartNode()
        endNode = branch.getEndNode()
        if self.isLeafNode(startNode):
            numLeafNodes += 1

        if self.isLeafNode(endNode):
            numLeafNodes += 1
        return numLeafNodes

    def _getBranchNodeFromNode(self, node: Node):
        """
        Internal Method, which is not recommended for external use.
        Returns the BranchNode corresponding to a given node from the
        list of branchNodes.
        Args:
            node (node): Node for which the branchNode should be determined

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
                foundNodes = self._findNextBranchOrLeafNodes(adjacentNode, thisNode)
                if len(foundNodes) != 0:
                    for node in foundNodes:
                        if node != previousNode:
                            nextBranchOrLeafNodes.append(node)
        return nextBranchOrLeafNodes

    def findNextBranchORLeafNodeBFS(self, thisNode):
        nextBranchOrLeafNodes = []
        visited = set()
        queue = []
        queue.append(thisNode)
        visited.add(thisNode)

        while len(queue) > 0:
            currentNode = queue.pop(0)
            adjacentNodes = currentNode.getAdjacentNodes()
            for adjacentNode in adjacentNodes:
                if adjacentNode not in visited:
                    if (
                        adjacentNode.getNumEdges() == 1
                        or adjacentNode.getNumEdges() > 2
                    ):
                        nextBranchOrLeafNodes.append(adjacentNode)
                    else:
                        queue.append(adjacentNode)
                    visited.add(adjacentNode)
                else:
                    pass
        return nextBranchOrLeafNodes

    def getLeafNodeFromNode(self, node: Node):
        if node in self._getLeafNodesAsNodes():
            return self.leafNodes[self._getLeafNodesAsNodes().index(node)]
        else:
            return None

    def getBranchNodeFromNode(self, node: Node):
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

    def getNodeIndex(self, node: Node):
        return self.nodes.index(node)

    def getBranchesFromNode(self, node: Node):
        branchList = []
        for branch in self.branches:
            if node in branch.getNodes():
                branchList.append(branch)
        return branchList

    def getBranchFromEdge(self, edge: Edge):
        """retuns the branch containing a given edge

        Args:
            edge (Edge): edge in the topologyModel

        Raises:
            ValueError: If edge corresponds to more than one branch.

        Returns:
            branch: branch the edge corresponds to.
        """  #
        branchList = []
        for branch in self.branches:
            if edge in branch.getEdges():
                branchList.append(branch)
        if len(branchList) > 1:
            for branch in branchList:
                print("Edge ifound in {}".format(branch.getName()))
            raise ValueError(
                "Edge found in more than one branch. Something went wrong."
            )
        return branchList[0]

    def getBranchIndexFromEdge(self, edge: Edge):
        """retuns the branch index containing a given edge

        Args:
            edge (Edge): edge in the topologyModel

        Raises:
            ValueError: If edge corresponds to more than one branch.

        Returns:
            branch: branch index the edge corresponds to.
        """
        return self.getBranchIndex(self.getBranchFromEdge(edge))

    def isLeafNode(self, node: Node):
        if node in self._getLeafNodesAsNodes():
            return True
        else:
            return False

    def isBranchNode(self, node: Node):
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

    def getAdjacentBranches(self, branch: branch):
        """
        Returns the adjacent branches to this branch.
        Sibling branches are branches which share a start or end node with the given branch.
        """
        adjancentBranches = []
        adjacentBranchCandidates = self.branches.copy()
        adjacentBranchCandidates.remove(branch)
        branchEnds = [branch.getStartNode(), branch.getEndNode()]
        for adjacentBranchCandidate in adjacentBranchCandidates:
            adjacentBranchCandidateEnds = [
                adjacentBranchCandidate.getStartNode(),
                adjacentBranchCandidate.getEndNode(),
            ]
            if any(
                branchEnd in branchEnds for branchEnd in adjacentBranchCandidateEnds
            ):
                adjancentBranches.append(adjacentBranchCandidate)
        return adjancentBranches

    def getBranchLength(self, branchNumber: int):
        return self.getBranch(branchNumber).getBranchInfo()["length"]

    def getLocalCoordinateFromBranchNode(self, branch: branch, node: Node):
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

    def getChildBranches(self, branch: branch):
        childBranches = []
        for childBranchCandidate in self.branches:
            if childBranchCandidate.getStartNode() == branch.getEndNode():
                childBranches.append(childBranchCandidate)
        return childBranches

    def getParentBranch(self, branch: branch):
        for parentBranchCandidate in self.branches:
            if parentBranchCandidate.getEndNode() == branch.getStartNode():
                parentBranch = parentBranchCandidate
        return parentBranch
