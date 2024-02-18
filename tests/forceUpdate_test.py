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
method = "StablePD"  # PD, LinearizedFeedback, StablePD
Kp = 300  # 0.1
Kd = 100  # 0.01


class TrackingWorldNode(dart.gui.osg.RealTimeWorldNode):
    def __init__(self, world, skel, Kp=Kp, Kd=Kd):
        super(TrackingWorldNode, self).__init__(world)
        self.skel = skel
        self.ForceUpdate = ForceUpdate(skel.clone(), Kp, Kd)
        world.addSkeleton(self.skel)

    def customPreStep(self):
        q = self.skel.getPositions()
        q_dot = self.skel.getVelocities()
        q_ddot = self.skel.getAccelerations()
        qd = np.zeros(self.skel.getNumDofs())
        qd[0] = 1
        qd[3] = -1
        qd[6] = 1
        qd[9] = -1
        qd[15] = 0
        qd_dot = np.zeros(self.skel.getNumDofs())
        qd_ddot = np.zeros(self.skel.getNumDofs())
        tauExt = self.ForceUpdate.computeExternalForceUpdateInGeneralizedCoordinates(
            q, q_dot, q_ddot, qd, qd_dot, qd_ddot, skel=self.skel, method=method
        )
        self.skel.setForces(tauExt)


def testForceController():
    # create world
    world = dart.simulation.World()
    # create dlo
    dlo = DeformableLinearObject(
        10,
        density=10,
        bendingStiffness=0,
        torsionalStiffness=0,
        bendingDampingCoeffs=0,
        torsionalDampingCoeffs=0,
    )
    # setup viewer
    viewer = dart.gui.osg.Viewer()
    grid = dart.gui.osg.GridVisual()
    grid.setPlaneType(dart.gui.osg.GridVisual.PlaneType.XY)
    grid.setOffset([0, 0, 0])
    viewer.addAttachment(grid)
    viewer.setUpViewInWindow(0, 0, 300, 300)
    viewer.setCameraHomePosition([8.0, 8.0, 4.0], [0, 0, -2.5], [0, 0, 1])
    # setup simulation
    node = TrackingWorldNode(world, dlo.skel.clone())
    viewer.addWorldNode(node)

    while True:
        node.customPreRefresh()
        if visualize:
            viewer.frame()
        node.customPreStep()
        world.step()
        node.customPostStep()


if __name__ == "__main__":
    testForceController()
