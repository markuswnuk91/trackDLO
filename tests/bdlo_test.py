# tests dart installation
import os, sys
import dartpy as dart
import time as time

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

    viewer.setUpViewInWindow(0, 0, 300, 200)
    viewer.setCameraHomePosition([8.0, 8.0, 4.0], [0, 0, -2.5], [0, 0, 1])
    # while True:
    #     viewer.frame()
    #     time.sleep(0.01)
    i = 0
    while True:
        world.step()
        if i % 2 == 0:
            viewer.frame()
        print(i)
        i += 1


if __name__ == "__main__":
    make_DLO()
