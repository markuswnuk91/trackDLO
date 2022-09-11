from cgi import test
import os, sys

try:
    sys.path.append(os.getcwd().replace("/tests", ""))
    from src.simulation.bdlo import node
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

    # def addChildNode(self, node):
    #     self.childNodes.append(node)


if __name__ == "__main__":
    testNode()
