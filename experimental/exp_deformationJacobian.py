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


class JacobianWorldNode(dart.gui.osg.RealTimeWorldNode):
    def __init__(self, world, skel, graspBodyNode):
        super(JacobianWorldNode, self).__init__(world)
        self.skel = skel
        world.addSkeleton(self.skel)

    def customPreStep(self, force):
        self.skel.getBodyNode(graspBodyNodeIndex).setExtForce(force)


def getDeformationJacobian(bdloModel, graspBodyNodeIndex, dx):
    # create world
    world = dart.simulation.World()
    # setup viewer
    viewer = dart.gui.osg.Viewer()
    grid = dart.gui.osg.GridVisual()
    grid.setPlaneType(dart.gui.osg.GridVisual.PlaneType.XY)
    grid.setOffset([0, 0, 0])
    viewer.addAttachment(grid)
    viewer.setUpViewInWindow(0, 0, 300, 300)
    viewer.setCameraHomePosition([8.0, 8.0, 4.0], [0, 0, -2.5], [0, 0, 1])

    # get current position of the grasped node
    graspBodyNode = bdloModel.skel.getBodyNode(graspBodyNodeIndex)
    targetFrame = dart.dynamics.SimpleFrame(
        dart.dynamics.Frame.World(), "target", dart.math.Isometry3()
    )
    # compute delta
    initalPosition = (
        bdloModel.skel.getBodyNode(graspBodyNodeIndex).getTransform().translation()
    )

    # get initial bodyNodePositions
    N = bdloModel.skel.getNumBodyNodes()
    initalBodyNodePositions = np.zeros((N, 3))
    for n in range(0, N):
        initalBodyNodePositions[n, :] = (
            bdloModel.skel.getBodyNode(n).getTransform().translation()
        )

    # setup simulation
    node = JacobianWorldNode(world, bdloModel.skel, graspBodyNode)
    viewer.addWorldNode(node)
    iteration = 0
    force = (1 / np.linalg.norm(dx)) * dx * bdloModel.density / 100
    amplificationFactor = 1.1
    delta = 0
    while delta < np.linalg.norm(dx):
        currentPosition = (
            bdloModel.skel.getBodyNode(graspBodyNodeIndex).getTransform().translation()
        )
        force = amplificationFactor**iteration * force
        if visualize:
            viewer.frame()
        node.customPreStep(force)
        world.step()
        delta = np.linalg.norm(initalPosition - currentPosition)
    finalBodyNodePosiitons = np.zeros((N, 3))
    for n in range(0, N):
        finalBodyNodePosiitons[n, :] = (
            bdloModel.skel.getBodyNode(n).getTransform().translation()
        )
    J = finalBodyNodePosiitons - initalBodyNodePositions
    return J


if __name__ == "__main__":
    # create dlo
    dlo = DeformableLinearObject(10, gravity=False)
    dx = np.array([0, 0.1, 0.0])
    graspBodyNodeIndex = 5
    J = getDeformationJacobian(dlo, graspBodyNodeIndex, dx)
    print(J)
