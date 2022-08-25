#tests dart installation
import dartpy as dart

#test visualization
testVis = False

def run_Dartpy():
        world = dart.utils.SkelParser.readWorld("dart://sample/skel/cubes.skel")
        world.setGravity([0, -9.81, 0])

        node = dart.gui.osg.RealTimeWorldNode(world)

        if testVis:
                viewer = dart.gui.osg.Viewer()
                viewer.addWorldNode(node)

                viewer.setUpViewInWindow(0, 0, 640, 480)
                viewer.setCameraHomePosition([0.8, 0.0, 0.8], [0, -0.25, 0], [0, 0.5, 0])
                viewer.frame()
        else:
                world.step()
        return run_Dartpy

def test_Dartpy():
    run_Dartpy()
    assert True