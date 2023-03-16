# tests dart installation
import os, sys
import dartpy as dart
import time as time

try:
    sys.path.append(os.getcwd().replace("/tests", ""))
    from src.simulation.bdlo import BranchedDeformableLinearObject
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


def test_ArenaWireHarness():

    testBDLO = BranchedDeformableLinearObject(
        **{"adjacencyMatrix": topologyGraph_ArenaWireHarness}
    )
    if vis:
        runViewer(testBDLO.skel)


if __name__ == "__main__":
    test_ArenaWireHarness()
