import os, sys
import numpy as np

try:
    sys.path.append(os.getcwd().replace("/tests", ""))
    from src.simulation.bdlo import node, edge, branch, topologyGraph
except:
    print("Imports for CPD failed.")
    raise


def testNode():
    testRootNode = node()
    testChildNode = node(testRootNode, name="testChildNode")
    testEdgeInfo = {"length": 1}
    testEdgeInfoNode = node(
        name="testEdgeInfoNode", parentNode=testChildNode, edgeInfo=testEdgeInfo
    )
    testNodeInfo = {"correspondingDartIndex": 0}
    testNodeInfoNode = node(
        parentNode=testChildNode, edgeInfo=None, nodeInfo=testNodeInfo
    )

    # test naming
    assert testRootNode.getName() == "Node_0"
    assert testChildNode.getName() == "testChildNode"
    assert testNodeInfoNode.getName() == "Node_3"

    # test getParent
    assert testRootNode.getParent() == None
    assert testChildNode.getParent() == testRootNode

    # test getChild
    assert testRootNode.getChilds()[0] == testChildNode

    # test getNodeInfo
    assert testNodeInfoNode.getNodeInfo()["correspondingDartIndex"] == 0

    # test EdgeInfo
    # assert testEdgeInfoNode.getParentEdge()[]

    # test getParentEdge
    assert testChildNode.getParentEdge().getParentNode() == testRootNode

    # test getChildEdges
    assert len(testChildNode.getChildEdges()) == 2
    assert testChildNode.getChildEdges()[0].getChildNode() == testEdgeInfoNode
    assert testChildNode.getChildEdges()[1].getChildNode() == testNodeInfoNode
    assert testChildNode.getChildEdges()[0].getParentNode() == testChildNode
    assert testChildNode.getChildEdges()[1].getParentNode() == testChildNode

    # test hasParentNode
    assert testRootNode.hasParentNode() == False
    assert testChildNode.hasParentNode() == True

    # test getNumChildNodes
    assert testRootNode.getNumChilds() == 1
    assert testChildNode.getNumChilds() == 2

    # test getNumChildNodes
    assert testEdgeInfoNode.getEdgeInfo(0) == testEdgeInfo
    assert testChildNode.getEdgeInfo(1) == testEdgeInfo
    # def addChildNode(self, node):
    #     self.childNodes.append(node)


def testBranch():
    firstBranchNode = node(name="firstBranchNode")
    secondBranchNode = node(parentNode=firstBranchNode, name="secondBranchNode")
    thirdBranchNode = node(parentNode=secondBranchNode, name="thirdBranchNode")
    firstBranch = branch(
        startNode=firstBranchNode, endNode=thirdBranchNode, name="firstBranch"
    )
    assert firstBranch.getName() == "firstBranch"
    assert firstBranch.getStartNode() == firstBranchNode
    assert firstBranch.getEndNode() == thirdBranchNode
    assert firstBranch.getMemberNodes()[0] == secondBranchNode
    assert firstBranch.getNodes() == [
        firstBranchNode,
        secondBranchNode,
        thirdBranchNode,
    ]
    assert firstBranch.getNumNodes() == 3
    assert firstBranch.getNumEdges() == 2

    secondBranch = branch(5)
    assert secondBranch.getName() == "Branch_1"
    assert secondBranch.getNumNodes() == 5


def testTopologyGraph():
    testGraph = np.array([[0, 5, 5, 0], [5, 0, 0, 0], [5, 0, 0, 5], [0, 0, 5, 0]])
    testTopology = topologyGraph(testGraph)
    assert testTopology.getNumNodes() == 4
    assert testTopology.getNodes()[0].getEdgeInfo(1)["length"] == 5
    assert testTopology.getNumBranches() == 2
    assert testTopology.getBranches()[0].getBranchInfo()["branchLength"] == 5
    assert testTopology.getBranches()[1].getBranchInfo()["branchLength"] == 10

    testGraphBranched = np.array(
        [
            [0, 3, 3, 3, 0, 0, 0, 0, 0],
            [3, 0, 0, 0, 3, 3, 0, 0, 0],
            [3, 0, 0, 0, 0, 0, 3, 3, 0],
            [3, 0, 0, 0, 0, 0, 0, 0, 3],
            [0, 3, 0, 0, 0, 0, 0, 0, 0],
            [0, 3, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 3, 0, 0, 0, 0, 0, 0],
            [0, 0, 3, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 3, 0, 0, 0, 0, 0],
        ]
    )
    testTopologyBranched = topologyGraph(testGraphBranched)
    assert testTopologyBranched.getNumNodes() == 9
    assert testTopologyBranched.getNumBranchNodes() == 3
    assert testTopologyBranched.getNumLeafNodes() == 5

    testGraphSingleDLO = np.array(
        [
            [0, 3, 0],
            [3, 0, 3],
            [0, 3, 0],
        ]
    )
    singleDLOTopology = topologyGraph(testGraphSingleDLO)
    assert singleDLOTopology.getNumNodes() == 3
    assert singleDLOTopology.getNumBranches() == 1
    assert singleDLOTopology.getNumBranchNodes() == 0
    assert singleDLOTopology.getNumLeafNodes() == 2


if __name__ == "__main__":
    testNode()
    testBranch()
    testTopologyGraph()
