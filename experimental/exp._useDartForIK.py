import os, sys
import dartpy as dart
import numpy as np
import time as time
from pytest import approx

try:
    sys.path.append(os.getcwd().replace("/tests", ""))
    from src.simulation.dlo import DeformableLinearObject
    from src.simulation.forceUpdate import ForceUpdate
except:
    print("Imports for DLO failed.")
    raise

visualize = True


def testDartIK():
    dlo = DeformableLinearObject(10)
    ikskel = dlo.skel.clone()
    # collect inverse kinematic modules and target positions
    ikModules = []
    targetFrames = []
    for i in range(0, ikskel.getNumBodyNodes()):
        ikModule = ikskel.getBodyNode(i).getOrCreateIK()
        ikModules.append(ikModule)
        targetFrame = ikModule.getTarget()
        targetFrame.setShape(dart.dynamics.SphereShape(0.01))
        visual = targetFrame.createVisualAspect()
        visual.setColor([1, 0, 0])

        targetFrames.append(targetFrame)
    # add offset to target frames
    offsets = np.zeros((len(targetFrames), 3))
    offsets[:, 0] = np.linspace(0, 1, len(targetFrames)) ** 2
    offsets[:, 2] = -1 * np.linspace(0, 0.1, len(targetFrames))

    for i, targetFrame in enumerate(targetFrames):
        currentPosition = targetFrame.getTransform().translation()
        targetFrame.setTranslation(currentPosition + offsets[i, :])

    for i, ikModule in enumerate(ikModules):
        ikModule.setTarget(targetFrames[i])
        ikModule.getErrorMethod().setAngularErrorWeights([0.1, 0.1, 0.1])
        if i >= 1:
            ikModules[i].setHierarchyLevel(1)
    ikModules[0].setHierarchyLevel(1)
    ikModules[-1].setHierarchyLevel(1)
    # targetFrames[0].setTranslation(
    #     targetFrames[0].getTransform().translation() + [0.1, 0.1, 0.2]
    # )
    # ikModules[0].setTarget(targetFrames[0])
    # solve IK for last body

    # ikModule = ikskel.getBodyNode(0).getOrCreateIK()
    # targetFrame = ikModule.getTarget()
    # targetFrame.setTranslation(
    #     targetFrame.getTransform().translation() + [0.0, 0.0, -0.1]
    # )
    # ikModule.setHierarchyLevel(1)
    # ikModule = ikskel.getBodyNode(5).getOrCreateIK()
    # targetFrame = ikModule.getTarget()
    # targetFrame.setTranslation(
    #     targetFrame.getTransform().translation() + [0.0, 0.5, 0.0]
    # )
    # ikModule.setHierarchyLevel(1)
    # ikModule = ikskel.getBodyNode(9).getOrCreateIK()
    # targetFrame = ikModule.getTarget()
    # targetFrame.setTranslation(
    #     targetFrame.getTransform().translation() + [0.0, 0.0, 1.1]
    # )
    wholeBodyIK = ikskel.getIK(True)
    wholeBodyIK.solveAndApply()
    q = ikskel.getPositions()
    print(q)
    # # set new target position
    # ikModules[1].setTarget(targetFrames[1])
    # ikModules[-1].solveAndApply()
    # q = ikskel.getPositions()
    # print(q)

    if visualize:
        skel = dlo.skel
        skel.setPositions(q)
        # setup simulation
        world = dart.simulation.World()
        viewer = dart.gui.osg.Viewer()
        grid = dart.gui.osg.GridVisual()
        grid.setPlaneType(dart.gui.osg.GridVisual.PlaneType.XY)
        grid.setOffset([0, 0, 0])
        viewer.addAttachment(grid)
        viewer.setUpViewInWindow(0, 0, 300, 300)
        viewer.setCameraHomePosition([8.0, 8.0, 4.0], [0, 0, -2.5], [0, 0, 1])
        node = dart.gui.osg.RealTimeWorldNode(world)
        viewer.addWorldNode(node)

        # add skelton and target frames
        world.addSkeleton(skel)
        for target in targetFrames:
            world.addSimpleFrame(target)
        while True:
            viewer.frame()


if __name__ == "__main__":
    testDartIK()
# # get the IK for a bodyNode
# inverseKinematics = skel.getBodyNode(0).getOrCreateIK()
# # get the current target
# targetFrame = inverseKinematics.getTarget()
# # set a new target
# newTargetFrame = targetFrame.clone()
# desiredPosition = np.array(([1, 1, 1]))
# newTargetFrame.setTranslation(desiredPosition)
# inverseKinematics.setTarget(newTargetFrame)
# # solveIK
# inverseKinematics.solveAndApply()
# # retrieve the solution
# q = skel.getPositions()
