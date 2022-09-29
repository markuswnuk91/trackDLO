import os, sys
import dartpy as dart
import numpy as np
import math
import time as time
from pytest import approx

try:
    sys.path.append(os.getcwd().replace("/tests", ""))
    from src.simulation.dlo import DeformableLinearObject
except:
    print("Imports for DLO failed.")
    raise


def test_appendDLO():
    world = dart.simulation.World()
    node = dart.gui.osg.RealTimeWorldNode(world)

    # test constructor from numSegments
    testDLO1 = DeformableLinearObject(5)
    testDLO2 = DeformableLinearObject(13)

    connectionJoint_prop = dart.dynamics.BallJointProperties()
    connectionJoint_prop.mRestPositions = np.array([math.pi / 2, 0, 0])
    connectionJoint_prop.mSpringStiffnesses = np.ones(3)

    testDLO2.skel.getBodyNode(0).moveTo(
        testDLO1.skel, testDLO1.skel.getBodyNode(3), connectionJoint_prop
    )

    testDLO1.skel.getBodyNode(4).setRestPosition
    # test simulation
    world.addSkeleton(testDLO1.skel)
    # Create world node and add it to viewer
    viewer = dart.gui.osg.Viewer()
    viewer.addWorldNode(node)

    # Grid settings
    grid = dart.gui.osg.GridVisual()
    grid.setPlaneType(dart.gui.osg.GridVisual.PlaneType.XY)
    grid.setOffset([0, 0, 0])
    viewer.addAttachment(grid)

    viewer.setUpViewInWindow(0, 0, 1200, 900)
    viewer.setCameraHomePosition([8.0, 8.0, 4.0], [0, 0, -2.5], [0, 0, 1])
    viewer.run()


if __name__ == "__main__":
    test_appendDLO()
