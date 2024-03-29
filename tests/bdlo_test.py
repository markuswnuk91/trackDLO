# tests dart installation
import os, sys
import dartpy as dart
import numpy as np
import math
import time as time
from pytest import approx
from unittest import TestCase
import numbers

try:
    sys.path.append(os.getcwd().replace("/tests", ""))
    from src.simulation.bdlo import BDLOTopology, BranchedDeformableLinearObject
    from src.modelling.topologyTemplates import topologyGraph_ArenaWireHarness
except:
    print("Imports for BDLO testing failed.")
    raise

vis = True


def runViewer(dartSkel):
    world = dart.simulation.World()
    node = dart.gui.osg.WorldNode(world)
    # Create world node and add it to viewer
    viewer = dart.gui.osg.Viewer()
    viewer.addWorldNode(node)

    # add skeleton
    world.addSkeleton(dartSkel)

    # Grid settings
    grid = dart.gui.osg.GridVisual()
    grid.setPlaneType(dart.gui.osg.GridVisual.PlaneType.XY)
    grid.setOffset([0, 0, 0])
    viewer.addAttachment(grid)

    viewer.setUpViewInWindow(0, 0, 1200, 900)
    viewer.setCameraHomePosition([8.0, 8.0, 4.0], [0, 0, -2.5], [0, 0, 1])
    viewer.run()


testTopologyModel_1 = np.array([[0, 1, 1, 0], [1, 0, 0, 0], [1, 0, 0, 1], [0, 0, 1, 0]])
testTopologyModel_2 = 0.1 * np.array(
    # [
    #     [0, 3, 3, 3, 0, 0, 0, 0, 0],
    #     [3, 0, 0, 0, 3, 3, 0, 0, 0],
    #     [3, 0, 0, 0, 0, 0, 3, 3, 0],
    #     [3, 0, 0, 0, 0, 0, 0, 0, 3],
    #     [0, 3, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 3, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 3, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 3, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 3, 0, 0, 0, 0, 0],
    # ]
    [
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    ]
)
testTopologyModel_3 = np.array([[0, 1], [1, 0]])

testTolologyModel_ICRA = np.array(
    [
        [0, 0.175, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0.175, 0, 0.08, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0.08, 0, 0.07, 0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0.07, 0, 0.07, 0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0.07, 0, 0.035, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.255, 0],
        [0, 0, 0, 0, 0.035, 0, 0.088, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0.088, 0, 0.032, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0.032, 0, 0, 0, 0, 0, 0.105, 0, 0.105, 0, 0, 0, 0],
        [0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0.155, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.155, 0, 0.09, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.09, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0.105, 0, 0, 0, 0, 0, 0.065, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.065, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0.105, 0, 0, 0, 0, 0, 0, 0, 0.225, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.225, 0, 0.1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0, 0, 0],
        [0, 0, 0, 0, 0.255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.19],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.19, 0],
    ]
)


def test_bdloTopology_1():

    testBdloSpec = BDLOTopology(**{"adjacencyMatrix": testTopologyModel_1})
    assert testBdloSpec.getNumNodes() == 4
    assert testBdloSpec.getNumBranches() == 1
    assert testBdloSpec.getBranches()[0].getBranchInfo()["length"] == 3
    for i, branch in enumerate(testBdloSpec.getBranches()):
        assert (
            testBdloSpec.getBranchSpec(i)
            == testBdloSpec.getBranches()[i].getBranchInfo()
        )


def test_TopologyGeneration_SimpleTopology():
    testBDLO_1 = BranchedDeformableLinearObject(
        **{"adjacencyMatrix": testTopologyModel_1, "name": "Test_BDLO_1"}
    )

    # test number of segmets is as specified
    testBDLO_1.branchSpecs[0]["numSegments"] == testBDLO_1.getNumSegments()
    if vis:
        runViewer(testBDLO_1.skel)


def test_TopologyGeneration_BranchedTopology():
    # testSpec_2 = BDLOTopology(testTopologyModel_2)
    testBDLO_2 = BranchedDeformableLinearObject(
        **{
            "adjacencyMatrix": testTopologyModel_2,
            "name": "Test_BDLO_2",
        }
    )
    if vis:
        runViewer(testBDLO_2.skel)


def test_TopologyGeneration_ElementaryGraph():
    testBDLO_3 = BranchedDeformableLinearObject(
        **{"adjacencyMatrix": testTopologyModel_3}
    )
    testBDLO_3.skel.setPosition(3, 0.4)
    print("Elemetary topology has: {} branches".format(testBDLO_3.getNumBranches()))
    print("Elemetary topology has: {} segments".format(testBDLO_3.getNumBodyNodes()))
    for i, branch in enumerate(testBDLO_3.getBranches()):
        print(
            "Corresponding bodyNodes for branch {} are: {}".format(
                i, testBDLO_3.getBodyNodeIndicesFromBranch(branch)
            )
        )
    if vis:
        runViewer(testBDLO_3.skel)


def test_TopologyGeneration_ICRATopology():
    testBDLO_ICRA = BranchedDeformableLinearObject(
        **{"adjacencyMatrix": testTolologyModel_ICRA}
    )
    testBDLO_ICRA.skel.setPosition(3, 0.5)
    for i, branch in enumerate(testBDLO_ICRA.getBranches()):
        print(
            "Corresponding bodyNodes for branch {} are: {}".format(
                i, testBDLO_ICRA.getBodyNodeIndicesFromBranch(branch)
            )
        )
    # check number of degrees of freedom match number of bodyNodes
    assert 3 * testBDLO_ICRA.getNumBodyNodes() + 3 == len(
        testBDLO_ICRA.skel.getPositions()
    )

    # # test branch lengths
    # assert testBDLO_ICRA.topology.getBranch(0).getBranchInfo()["length"] == 0.255
    # assert testBDLO_ICRA.topology.getBranch(1).getBranchInfo()["length"] == 0.07
    # assert testBDLO_ICRA.topology.getBranch(2).getBranchInfo()["length"] == 0.1
    # assert testBDLO_ICRA.topology.getBranch(3).getBranchInfo()["length"] == 0.07
    # assert testBDLO_ICRA.topology.getBranch(4).getBranchInfo()["length"] == 0.345
    # assert testBDLO_ICRA.topology.getBranch(5).getBranchInfo()["length"] == 0.155
    # assert testBDLO_ICRA.topology.getBranch(6).getBranchInfo()["length"] == 0.445
    # assert testBDLO_ICRA.topology.getBranch(7).getBranchInfo()[
    #     "length"
    # ] - 0.170 == approx(0, abs=1e-8)
    # assert testBDLO_ICRA.topology.getBranch(8).getBranchInfo()[
    #     "length"
    # ] - 0.43 == approx(0, abs=1e-8)

    # test taht branch points are located correctly

    # assert (
    #     testBDLO_ICRA.topology.getBranchLength(0)
    #     - testBDLO_ICRA.getSegmentLengthFromBranch(0) / 2
    #     == testBDLO_ICRA.getBranchPointBodyNodes()[0]
    #     .getWorldTransform()
    #     .translation()[2]
    # )

    # # test getBranchBodyNodes
    # assert testBDLO_ICRA.getBranchBodyNodes(0)[0] == testBDLO_ICRA.skel.getBodyNode(0)

    # # test getBranchBodyNodeIndices
    # assert testBDLO_ICRA.getBranchBodyNodeIndices(0)[0] == 0

    # # test getLeafBodyNodes
    # # assert testBDLO_ICRA.getLeafBodyNodes()[0] == testBDLO_ICRA.skel.getBodyNode(0)
    # # assert testBDLO_ICRA.getLeafBodyNodeIndices()[0] == 0

    # # test getBranchPointBodyNodes
    # assert len(testBDLO_ICRA.getBranchPointBodyNodes()) == 4
    # assert len(testBDLO_ICRA.getBranchPointBodyNodeIndices()) == 4

    # # test getBranchIndexFromBodyNode
    # assert testBDLO_ICRA.getBranchIndexFromBodyNodeIndex(0) == 0

    # test indexes for memberNodes
    # assert testBDLO_ICRA.topology.getBranch(0).getMemberNodes().getNodeInfo()

    # test getBranchBodyNodes
    if vis:
        runViewer(testBDLO_ICRA.skel)


def test_TopologyGeneration_ArenaTopology():
    testBDLO = BranchedDeformableLinearObject(
        **{"adjacencyMatrix": topologyGraph_ArenaWireHarness}
    )

    # test BDLO functions
    testBDLO.getBranchRootBodyNodeIndex(0)
    testBDLO.getBranchRootDofIndices(0)
    testBDLO.setBranchRootDof(1, 0, np.pi * 3 / 4)
    testBDLO.setBranchRootDofs(2, np.array([0, 0, 0]))
    testBDLO.setBranchRootDofs(3, np.array([-np.pi * 3 / 4, 0, 0]))
    testBDLO.setBranchRootDofs(4, np.array([0, 0, 0]))
    testBDLO.setBranchRootDofs(5, np.array([np.pi / 4, 0, 0]))
    testBDLO.setBranchRootDofs(6, np.array([0, 0, 0]))

    if vis:
        runViewer(testBDLO.skel)


def test_CartesianPositionFunctions():
    testBDLO = BranchedDeformableLinearObject(
        **{"adjacencyMatrix": topologyGraph_ArenaWireHarness}
    )
    testBDLO.setBranchRootDof(1, 0, np.pi * 3 / 4)
    testBDLO.setBranchRootDofs(2, np.array([0, 0, 0]))
    testBDLO.setBranchRootDofs(3, np.array([-np.pi * 3 / 4, 0, 0]))
    testBDLO.setBranchRootDofs(4, np.array([0, 0, 0]))
    testBDLO.setBranchRootDofs(5, np.array([np.pi / 4, 0, 0]))
    testBDLO.setBranchRootDofs(6, np.array([0, 0, 0]))

    # testing
    branchIdx = 0
    S = np.linspace(0, 1, 11)
    for branchIdx in range(0, testBDLO.getNumBranches()):
        Xb_Test = testBDLO.getCaresianPositionsFromLocalCoordinates(branchIdx, S)
        for i, s in enumerate(S):
            x_Test = testBDLO.getCartesianPositionFromBranchLocalCoordinate(
                branchIdx, s
            )
            print("Position of branch {} at s={}: {}".format(branchIdx, s, x_Test))
            print("Difference is {}".format(Xb_Test[i] - x_Test))
        print(Xb_Test)


if __name__ == "__main__":
    # test modelSpecificaion
    # test_bdloTopology_1()

    # test topologyGeneration
    # test_TopologyGeneration_SimpleTopology()
    # test_TopologyGeneration_BranchedTopology()
    # test_TopologyGeneration_ElementaryGraph()
    # test_TopologyGeneration_ICRATopology()
    test_TopologyGeneration_ArenaTopology()
    test_CartesianPositionFunctions()
