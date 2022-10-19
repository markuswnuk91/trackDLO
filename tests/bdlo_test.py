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
    from src.simulation.bdlo import bdloSpecification, BranchedDeformableLinearObject
except:
    print("Imports for BDLO testing failed.")
    raise

visualize = True

testTopologyModel_1 = np.array([[0, 1, 1, 0], [1, 0, 0, 0], [1, 0, 0, 1], [0, 0, 1, 0]])
testTopologyModel_2 = 0.1 * np.array(
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

    testBdloSpec = bdloSpecification(testTopologyModel_1)
    assert testBdloSpec.getNumNodes() == 4
    assert testBdloSpec.getNodes()[0].getEdgeInfo(1)["length"] == 1
    assert testBdloSpec.getNumBranches() == 2
    assert testBdloSpec.getBranches()[0].getBranchInfo()["length"] == 1
    assert testBdloSpec.getBranches()[1].getBranchInfo()["length"] == 2

    for i, branch in enumerate(testBdloSpec.getBranches()):
        assert (
            testBdloSpec.getBranchSpec(i)
            == testBdloSpec.getBranches()[i].getBranchInfo()
        )


def test_branchedDeformableLinearObject():
    testSpec_1 = bdloSpecification(testTopologyModel_1)
    testBDLO_1 = BranchedDeformableLinearObject(testSpec_1, name="Test_BDLO_1")

    testSpec_2 = bdloSpecification(testTopologyModel_2)
    testBDLO_2 = BranchedDeformableLinearObject(testSpec_2, name="Test_BDLO_2")
    testBDLO_2.skel.setPosition(3, 0.2)

    testSpec_3 = bdloSpecification(testTopologyModel_3)
    testBDLO_3 = BranchedDeformableLinearObject(testSpec_3)
    testBDLO_3.skel.setPosition(3, 0.4)

    testSpec_ICRA = bdloSpecification(testTolologyModel_ICRA)
    testBDLO_ICRA = BranchedDeformableLinearObject(testSpec_ICRA, name="Test_BDLO_ICRA")
    testBDLO_ICRA.skel.setPosition(3, 0.5)
    # test branch lengths
    assert testBDLO_ICRA.topology.getBranch(0).getBranchInfo()["length"] == 0.255
    assert testBDLO_ICRA.topology.getBranch(1).getBranchInfo()["length"] == 0.07
    assert testBDLO_ICRA.topology.getBranch(2).getBranchInfo()["length"] == 0.1
    assert testBDLO_ICRA.topology.getBranch(3).getBranchInfo()["length"] == 0.07
    assert testBDLO_ICRA.topology.getBranch(4).getBranchInfo()["length"] == 0.345
    assert testBDLO_ICRA.topology.getBranch(5).getBranchInfo()["length"] == 0.155
    assert testBDLO_ICRA.topology.getBranch(6).getBranchInfo()["length"] == 0.445
    assert testBDLO_ICRA.topology.getBranch(7).getBranchInfo()[
        "length"
    ] - 0.170 == approx(0, abs=1e-8)
    assert testBDLO_ICRA.topology.getBranch(8).getBranchInfo()[
        "length"
    ] - 0.43 == approx(0, abs=1e-8)

    # test taht branch points are located correctly

    assert (
        testBDLO_ICRA.topology.getBranchLength(0)
        - testBDLO_ICRA.getSegmentLengthFromBranch(0) / 2
        == testBDLO_ICRA.getBranchPointBodyNodes()[0]
        .getWorldTransform()
        .translation()[2]
    )

    # test getBranchBodyNodes
    assert testBDLO_ICRA.getBranchBodyNodes(0)[0] == testBDLO_ICRA.skel.getBodyNode(0)

    # test getBranchBodyNodeIndices
    assert testBDLO_ICRA.getBranchBodyNodeIndices(0)[0] == 0

    # test getLeafBodyNodes
    assert testBDLO_ICRA.getLeafBodyNodes()[0] == testBDLO_ICRA.skel.getBodyNode(0)
    assert testBDLO_ICRA.getLeafBodyNodeIndices()[0] == 0

    # test getBranchPointBodyNodes
    assert len(testBDLO_ICRA.getBranchPointBodyNodes()) == 4
    assert len(testBDLO_ICRA.getBranchPointBodyNodeIndices()) == 4

    # test getBranchIndexFromBodyNode
    assert testBDLO_ICRA.getBranchIndexFromBodyNodeIndex(0) == 0

    # test getBranchBodyNodes
    if visualize:
        world = dart.simulation.World()
        node = dart.gui.osg.WorldNode(world)
        # Create world node and add it to viewer
        viewer = dart.gui.osg.Viewer()
        viewer.addWorldNode(node)

        # add skeleton
        world.addSkeleton(testBDLO_1.skel)
        world.addSkeleton(testBDLO_2.skel)
        world.addSkeleton(testBDLO_3.skel)
        world.addSkeleton(testBDLO_ICRA.skel)

        # Grid settings
        grid = dart.gui.osg.GridVisual()
        grid.setPlaneType(dart.gui.osg.GridVisual.PlaneType.XY)
        grid.setOffset([0, 0, 0])
        viewer.addAttachment(grid)

        viewer.setUpViewInWindow(0, 0, 1200, 900)
        viewer.setCameraHomePosition([8.0, 8.0, 4.0], [0, 0, -2.5], [0, 0, 1])
        viewer.run()


if __name__ == "__main__":
    test_bdloTopology_1()
    test_branchedDeformableLinearObject()
