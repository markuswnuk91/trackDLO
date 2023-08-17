import sys, os
import numpy as np
import dartpy as dart

try:
    sys.path.append(os.getcwd().replace("/src/visualization", ""))
except:
    print("Imports for dartViewer failed.")
    raise


class DartVisualizer(object):
    def __init__(self):
        self.world = dart.simulation.World()
        self.node = dart.gui.osg.WorldNode(self.world)
        # Create world node and add it to viewer
        self.viewer = dart.gui.osg.Viewer()
        self.viewer.addWorldNode(self.node)

        # Grid settings
        grid = dart.gui.osg.GridVisual()
        grid.setPlaneType(dart.gui.osg.GridVisual.PlaneType.XY)
        grid.setOffset([0, 0, 0])
        self.viewer.addAttachment(grid)

        self.viewer.setUpViewInWindow(0, 0, 1200, 900)
        self.viewer.setCameraHomePosition([8.0, 8.0, 4.0], [0, 0, -2.5], [0, 0, 1])

    def addSkeleton(self, skel):
        self.world.addSkeleton(skel)

    def run(self, visualize=True):
        while True:
            self.node.customPreRefresh()
            if visualize:
                self.viewer.frame()
            self.node.customPreStep()
            self.world.step()
            self.node.customPostStep()

    def setCameraPosition(self, eye=[8.0, 8.0, 4.0], center=[0, 0, -2.5], up=[0, 0, 1]):
        self.viewer.setCameraHomePosition(eye, center, up)

    def showFrame(self):
        self.viewer.frame()

    def computeInverseKinematicsWithCartesianForces(
        self,
        skel,
        targetPositions,
        kp=1,
        kd=0.1,
        cartesianErrorThreshold=2,
        velocityThreshold=0.3,
        vis=False,
        verbose=False,
    ):
        world = dart.simulation.World()
        node = dart.gui.osg.WorldNode(world)
        if vis:
            viewer = dart.gui.osg.Viewer()
            viewer.addWorldNode(node)
            grid = dart.gui.osg.GridVisual()
            grid.setPlaneType(dart.gui.osg.GridVisual.PlaneType.XY)
            grid.setOffset([0, 0, 0])
            viewer.addAttachment(grid)
            viewer.setUpViewInWindow(0, 0, 1200, 900)
            viewer.setCameraHomePosition([8.0, 8.0, 4.0], [0, 0, -2.5], [0, 0, 1])

        N = skel.getNumBodyNodes()
        world.addSkeleton(skel)
        currentPositions = np.zeros((N, 3))
        currentVeclocities = np.zeros((N, 3))
        summedVelocities = 0
        positionDifference = np.inf
        while (
            summedVelocities > velocityThreshold
            or positionDifference > cartesianErrorThreshold
        ):
            for bodyNodeIndex in range(0, skel.getNumBodyNodes()):
                bodyNode = skel.getBodyNode(bodyNodeIndex)
                currentPositions[
                    bodyNodeIndex, :
                ] = bodyNode.getTransform().translation()
                currentVeclocities[bodyNodeIndex, :] = bodyNode.getLinearVelocity()
                force = (
                    kp
                    * (
                        targetPositions[bodyNodeIndex, :]
                        - currentPositions[bodyNodeIndex, :]
                    )
                    - kd * currentVeclocities[bodyNodeIndex, :]
                )
                bodyNode.setExtForce(force, np.zeros((3,)))
            world.step()

            summedVelocities = np.sum(np.linalg.norm(currentVeclocities, axis=1))
            positionDifference = np.sum(
                np.linalg.norm(targetPositions - currentPositions, axis=1)
            )

            if verbose:
                print("Position difference: {}".format(positionDifference))
                print("Summed velocities: {}".format(summedVelocities))

            if vis:
                viewer.frame()
        return skel.getPositions()
