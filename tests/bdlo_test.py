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

testTopologyModel_1 = np.array([[0, 5, 5, 0], [5, 0, 0, 0], [5, 0, 0, 5], [0, 0, 5, 0]])


def test_bdloTopology():

    testBdloSpec = bdloSpecification(testTopologyModel_1)
    assert testBdloSpec.getNumNodes() == 4
    assert testBdloSpec.getNodes()[0].getEdgeInfo(1)["length"] == 5
    assert testBdloSpec.getNumBranches() == 2
    assert testBdloSpec.getBranches()[0].getBranchInfo()["length"] == 5
    assert testBdloSpec.getBranches()[1].getBranchInfo()["length"] == 10

    for i, branch in enumerate(testBdloSpec.getBranches()):
        assert (
            testBdloSpec.getBranchSpec(i)
            == testBdloSpec.getBranches()[i].getBranchInfo()
        )


def test_branchedDeformableLinearObject():
    testSpec = bdloSpecification(testTopologyModel_1)
    testBDLO_1 = BranchedDeformableLinearObject(testSpec, name="Test_BDLO")

    if visualize:
        world = dart.simulation.World()
        node = dart.gui.osg.WorldNode(world)
        # Create world node and add it to viewer
        viewer = dart.gui.osg.Viewer()
        viewer.addWorldNode(node)

        # add skeleton
        world.addSkeleton(testBDLO_1.skel)

        # Grid settings
        grid = dart.gui.osg.GridVisual()
        grid.setPlaneType(dart.gui.osg.GridVisual.PlaneType.XY)
        grid.setOffset([0, 0, 0])
        viewer.addAttachment(grid)

        viewer.setUpViewInWindow(0, 0, 1200, 900)
        viewer.setCameraHomePosition([8.0, 8.0, 4.0], [0, 0, -2.5], [0, 0, 1])
        viewer.run()


if __name__ == "__main__":
    test_bdloTopology()
    test_branchedDeformableLinearObject()
