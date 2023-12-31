import os, sys
import numpy as np

try:
    sys.path.append(os.getcwd().replace("/tests", ""))
    from src.modelling.topologyModel import node, edge, branch, topologyModel
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


def testtopologyModel():
    testGraph = np.array([[0, 5, 5, 0], [5, 0, 0, 0], [5, 0, 0, 5], [0, 0, 5, 0]])
    testTopology = topologyModel(testGraph)
    assert testTopology.getNumNodes() == 4
    assert testTopology.getNodes()[0].getEdgeInfo(1)["length"] == 5
    assert testTopology.getNumBranches() == 2
    assert testTopology.getBranches()[0].getBranchInfo()["length"] == 5
    assert testTopology.getBranches()[1].getBranchInfo()["length"] == 10
    assert testTopology.getSummedLength() == 15

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
    testTopologyBranched = topologyModel(testGraphBranched)
    assert testTopologyBranched.getNumNodes() == 9
    assert testTopologyBranched.getNumBranchNodes() == 3
    assert testTopologyBranched.getNumLeafNodes() == 5
    assert set(testTopologyBranched.getLeafNodeIndices()) == set([4, 5, 6, 7, 8])
    assert set(testTopologyBranched.getBranchNodeIndices()) == set([0, 1, 2])
    assert testTopologyBranched.getChildBranches(testTopologyBranched.getBranch(0)) == [
        testTopologyBranched.getBranch(3),
        testTopologyBranched.getBranch(4),
    ]
    # test isRootBranch
    assert testTopologyBranched.isRootBranch(testTopologyBranched.getBranch(0)) == True
    assert testTopologyBranched.isRootBranch(testTopologyBranched.getBranch(1)) == True
    assert testTopologyBranched.isRootBranch(testTopologyBranched.getBranch(2)) == True
    assert testTopologyBranched.isRootBranch(testTopologyBranched.getBranch(3)) == False
    assert testTopologyBranched.isRootBranch(testTopologyBranched.getBranch(4)) == False
    assert testTopologyBranched.isRootBranch(testTopologyBranched.getBranch(5)) == False
    assert testTopologyBranched.isRootBranch(testTopologyBranched.getBranch(6)) == False

    testGraphSingleDLO = np.array(
        [
            [0, 3, 0],
            [3, 0, 3],
            [0, 3, 0],
        ]
    )
    singleDLOTopology = topologyModel(testGraphSingleDLO)
    assert singleDLOTopology.getNumNodes() == 3
    assert singleDLOTopology.getNumBranches() == 1
    assert singleDLOTopology.getNumBranchNodes() == 0
    assert singleDLOTopology.getNumLeafNodes() == 2
    assert singleDLOTopology.getLeafNodeIndices() == [0, 2]
    assert (
        singleDLOTopology.getBranchesFromNode(singleDLOTopology[1])[0].getStartNode()
        == singleDLOTopology[0]
    )

    # test getChildBranches
    assert singleDLOTopology.getChildBranches(singleDLOTopology.getBranches()[0]) == []

    # test input checking against wrong adjacency matrix
    wrongTopologyGraph = np.array([[1, 0], [0, 1]])
    try:
        testWrongtopologyModel = topologyModel(wrongTopologyGraph)
        detectedWrongTopologyInput = False
    except:
        detectedWrongTopologyInput = True
    assert detectedWrongTopologyInput == True

    # test getLocalCoordinateFromBranchNode
    assert (
        singleDLOTopology.getLocalCoordinateFromBranchNode(
            singleDLOTopology.getBranch(0), singleDLOTopology.getNodes()[1]
        )
        == 0.5
    )


if __name__ == "__main__":
    testNode()
    testBranch()
    testtopologyModel()
