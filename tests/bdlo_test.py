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
