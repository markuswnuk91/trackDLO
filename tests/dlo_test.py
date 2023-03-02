# tests dlo implementation
import os, sys
import dartpy as dart
import numpy as np
import time as time
from pytest import approx

try:
    sys.path.append(os.getcwd().replace("/tests", ""))
    from src.simulation.dlo import DeformableLinearObject
except:
    print("Imports for DLO failed.")
    raise

visualize = True


def test_initDLO():
    world = dart.simulation.World()

    # test constructor from numSegments
    testDLO = DeformableLinearObject(13)
    assert testDLO.getNumSegments() == 13

    # test default constructor
    testDLO = DeformableLinearObject()
    assert testDLO.getNumSegments() == 20
    assert testDLO.density == 1000
    assert testDLO.radius == 0.01
    assert testDLO.length == 1
    assert testDLO.segmentLength == 1 / 20
    assert testDLO.stiffness == 1
    assert testDLO.damping == 0.1
    assert (testDLO.color == np.array([0, 0, 1])).all()
    assert testDLO.gravity == True
    assert testDLO.collidable == True
    assert testDLO.adjacentBodyCheck == False
    assert testDLO.enableSelfCollisionCheck == True
    assert (
        testDLO.skel.getBodyNode(0).getWorldTransform().translation()
        == np.array([0, 0, testDLO.segmentLength / 2])
    ).all()
    assert np.linalg.norm(
        testDLO.skel.getBodyNode(testDLO.skel.getNumBodyNodes() - 1)
        .getWorldTransform()
        .translation()
        - np.array([0, 0, testDLO.length - testDLO.segmentLength / 2])
    ) == approx(0, abs=1e-12)

    # test setInitialPosition
    testDLO.setInitialPosition(0, 2.0)
    assert testDLO.skel.getDof(0).getPosition() == 2.0
    assert testDLO.skel.getDof(0).getRestPosition() == 2.0
    assert testDLO.getPosition(0) == 2.0

    testDLO.setInitialPosition(1, 2.0)
    assert testDLO.skel.getDof(1).getPosition() == 2.0
    assert testDLO.skel.getDof(1).getRestPosition() == 2.0
    assert testDLO.getPosition(1) == 2.0

    testDLO.setInitialPosition(2, 2.0)
    assert testDLO.skel.getDof(2).getPosition() == 2.0
    assert testDLO.skel.getDof(2).getRestPosition() == 2.0
    assert testDLO.getPosition(2) == 2.0

    testDLO.setInitialPosition(3, 0.1)
    assert testDLO.skel.getDof(3).getPosition() == 0.1
    assert testDLO.skel.getDof(3).getRestPosition() == 0.1
    assert testDLO.getPosition(3) == 0.1

    testDLO.setInitialPosition(4, 0.1)
    assert testDLO.skel.getDof(4).getPosition() == 0.1
    assert testDLO.skel.getDof(4).getRestPosition() == 0.1
    assert testDLO.getPosition(4) == 0.1

    testDLO.setInitialPosition(5, 0.1)
    assert testDLO.skel.getDof(5).getPosition() == 0.1
    assert testDLO.skel.getDof(5).getRestPosition() == 0.1
    assert testDLO.getPosition(5) == 0.1

    # test masss and inertia values
    for i in range(testDLO.getNumSegments()):
        assert (
            testDLO.skel.getBodyNode(i).getInertia().getMass()
            == dart.dynamics.CylinderShape(
                testDLO.radius, testDLO.segmentLength - 2 * testDLO.radius
            ).getVolume()
            * testDLO.density
        )
        assert (
            testDLO.skel.getBodyNode(i).getInertia().getMoment()
            == dart.dynamics.CylinderShape(
                testDLO.radius, testDLO.segmentLength - 2 * testDLO.radius
            ).computeInertiaFromDensity(testDLO.density)
        ).all()

    # test adding a simple frame
    testDLO.addSimpleFrame(
        key="Target",
        bodyNodeNumber=0,
    )
    testTransform = np.array(
        (
            [1, 0, 0, testDLO.radius],
            [0, 1, 0, 2 * testDLO.radius],
            [0, 0, 1, testDLO.segmentLength / 2],
            [0, 0, 0, 1],
        )
    )
    testDLO.addSimpleFrame(
        key="Testframes",
        bodyNodeNumber=3,
        name="myTestFrame",
        relTransform=testTransform,
        shape=dart.dynamics.BoxShape(3 * testDLO.radius * np.ones(3)),
        shapeColor=np.array([1, 0, 0]),
    )

    testDLO.addSimpleFrame(
        key="Testframes",
        bodyNodeNumber=10,
        name="myTestFrame",
        shape=dart.dynamics.BoxShape(3 * testDLO.radius * np.ones(3)),
        shapeColor=np.array([1, 0, 0]),
    )

    # test simulation
    world.addSkeleton(testDLO.skel)
    node = dart.gui.osg.RealTimeWorldNode(world)
    try:
        world.step()
        stepError = False
    except:
        stepError = True

    assert not stepError

    if visualize:
        try:
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

            visError = False
        except:
            visError = True

        assert not visError


if __name__ == "__main__":
    test_initDLO()
