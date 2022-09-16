# tests dart installation
import dartpy as dart
import os, sys

try:
    sys.path.append(os.getcwd().replace("/tests", ""))
    from src.simulation.bdlo import DeformableLinearObject
except:
    print("Imports for DLO failed.")
    raise


def make_DLO():
    world = dart.simulation.World()
    myDLO = DeformableLinearObject()
    world.addSkeleton(myDLO.skel)

    node = dart.gui.osg.WorldNode(world)

    # Create world node and add it to viewer
    viewer = dart.gui.osg.Viewer()
    viewer.addWorldNode(node)

    # Grid settings
    grid = dart.gui.osg.GridVisual()
    grid.setPlaneType(dart.gui.osg.GridVisual.PlaneType.XY)
    grid.setOffset([0, 0, 0])
    viewer.addAttachment(grid)

    viewer.setUpViewInWindow(0, 0, 1280, 760)
    viewer.setCameraHomePosition([8.0, 8.0, 4.0], [0, 0, -2.5], [0, 0, 1])
    viewer.run()


if __name__ == "__main__":
    make_DLO()
