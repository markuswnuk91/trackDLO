import os, sys
import dartpy as dart
import numpy as np

try:
    sys.path.append(os.getcwd().replace("/tests", ""))
    from src.simulation.urdfLoader import URDFLoader
except:
    print("Imports for testing URDF Loader failed.")
    raise


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


def test_loadPanda():
    urdfLoader = URDFLoader()
    pandaSkel = urdfLoader.loadPandaAndGripper()
    runViewer(pandaSkel)


def test_loadCell():
    urdfLoader = URDFLoader()
    skel = urdfLoader.loadCell()
    runViewer(skel)


def test_loadClipBoard():
    urdfLoader = URDFLoader()
    skel = urdfLoader.loadClipBoard()
    runViewer(skel)


def test_loadFixture():
    urdfLoader = URDFLoader()
    skel = urdfLoader.loadFixture()
    runViewer(skel)


if __name__ == "__main__":
    # test_loadPanda()
    # test_loadCell()
    # test_loadClipBoard()
    test_loadFixture()
