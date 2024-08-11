import sys, os
import numpy as np
import dartpy as dart
import time
from warnings import warn
from scipy.spatial.transform import Rotation as R

try:
    sys.path.append(os.getcwd().replace("/src/visualization", ""))
    from src.simulation.urdfLoader import URDFLoader
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
        self.viewer.setCameraHomePosition([8.0, 8.0, 4.0], [0, 0, -2.5], [0, 0, 1])

        self.arrowList = []

    def setupViewer(
        self,
        worldNode,
        grid=True,
        eye=[8.0, 8.0, 4.0],
        center=[0, 0, -2.5],
        up=[0, 0, 1],
    ):
        viewer = dart.gui.osg.Viewer()
        viewer.addWorldNode(worldNode)
        if grid:
            # Grid settings
            grid = dart.gui.osg.GridVisual()
            grid.setPlaneType(dart.gui.osg.GridVisual.PlaneType.XY)
            grid.setOffset([0, 0, 0])
        viewer.addAttachment(grid)
        viewer.setCameraHomePosition(eye, center, up)
        return viewer

    def addSkeleton(self, skel):
        self.world.addSkeleton(skel)

    def addPointCloud(
        self,
        points,
        colors=None,
        size=None,
        alpha=None,
    ):
        """
        Visualize a PointCloud in DART
        """
        colors = [0, 0, 0] if colors is None else colors
        size = 0.005 if size is None else size
        alpha = 1 if alpha is None else alpha

        pointCloudShape = dart.dynamics.PointCloudShape(size)
        frame = dart.dynamics.SimpleFrame(
            dart.dynamics.Frame.World(), name="PointCloud"
        )
        frame.setShape(pointCloudShape)
        pointCloudShape.addPoint(points)
        visual = frame.createVisualAspect()

        if isinstance(colors, np.ndarray) and colors.shape[0] > 1:
            pointCloudShape.setColorMode(dart.dynamics.PointCloudShape.BIND_PER_POINT)
            createMyColorMap = np.ones((4, len(colors[0])))
            createMyColorMap[:3, :] = colors / 255
            pointCloudShape.setColors(createMyColorMap.T)
        else:
            visual.setColor(colors)
            visual.setAlpha(alpha)
        self.world.addSimpleFrame(frame)

    def addBox(
        self,
        sideLengths,
        offset=None,
        rotation=None,
        color=None,
    ):
        color = [0, 0, 0] if color is None else color
        offset = [0, 0, 0] if offset is None else offset
        rotation = np.eye(3) if rotation is None else rotation

        frame = dart.dynamics.SimpleFrame(dart.dynamics.Frame.World(), name="Box")
        boxShape = dart.dynamics.BoxShape(sideLengths)
        boxShape.addDataVariance(dart.dynamics.BoxShape.DYNAMIC_COLOR)
        frame.setShape(boxShape)

        frame.createVisualAspect()
        frame.getVisualAspect().setColor(color)

        frame.setRelativeTranslation(offset)
        frame.setRelativeRotation(rotation)
        self.world.addSimpleFrame(frame)
        return

    def addSphere(
        self,
        radius,
        offset=None,
        color=None,
    ):
        color = [0, 0, 0] if color is None else color
        offset = [0, 0, 0] if offset is None else offset

        frame = dart.dynamics.SimpleFrame(dart.dynamics.Frame.World(), name="Sphere")
        shape = dart.dynamics.SphereShape(radius)
        shape.addDataVariance(dart.dynamics.BoxShape.DYNAMIC_COLOR)
        frame.setShape(shape)

        frame.createVisualAspect()
        frame.getVisualAspect().setColor(color)

        frame.setRelativeTranslation(offset)
        self.world.addSimpleFrame(frame)
        return

    def addArrow(
        self,
        startPoint,
        endPoint,
        color=None,
        radius=None,
    ):
        radius = 0.001 if radius is None else radius
        color = [1, 0, 0] if color is None else color
        arrowName = "arrow_" + str(len(self.arrowList))
        arrowShapeProperties = dart.dynamics.ArrowShapeProperties(radius=radius)
        simpleFrame = dart.dynamics.SimpleFrame(
            dart.dynamics.Frame.World(), name=arrowName
        )
        simpleFrame.setShape(
            dart.dynamics.ArrowShape(
                startPoint,
                endPoint,
                arrowShapeProperties,
            )
        )
        visual = simpleFrame.createVisualAspect()
        visual.setColor(color)
        self.world.addSimpleFrame(simpleFrame)
        self.arrowList.append(simpleFrame)
        return simpleFrame

    def removeArrow(self, arrow):
        self.world.removeSimpleFrame(arrow)
        self.arrowList.remove(arrow)

    def removearrow(self, name):
        self.world.removeSimpleFrame()

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

    def showFrame(self, x=0, y=0, width=1200, height=900):
        if self.viewer is None:
            self.viewer.setUpViewInWindow(x, y, width, height)
        self.viewer.frame()

    def runVisualization(self, x=0, y=0, width=1200, height=900):
        self.viewer.setUpViewInWindow(x, y, width, height)
        self.viewer.run()

    def showConfiguration(self, skel, q):
        skel.setPositions(q)
        self.addSkeleton(skel)
        self.runVisualization()

    def saveFrame(
        self,
        savePath,
        grid=True,
        x=0,
        y=0,
        width=1200,
        height=900,
        eye=[8.0, 8.0, 4.0],
        center=[0, 0, -2.5],
        up=[0, 0, 1],
        format=".png",
    ):
        viewer = self.setupViewer(
            self.node,
            grid=True,
            eye=eye,
            center=center,
            up=up,
        )
        viewer.setUpViewInWindow(x, y, width, height)
        viewer.captureScreen(savePath + format)
        viewer.frame()
        time.sleep(1)
        # del viewer
        return

    def computeCartesianForceInverseKinematics(
        self,
        skel,
        targetPositions,
        qInit=None,
        numIterMax=1000,
        kp=1,
        kd=0.1,
        cartesianErrorThreshold=2,
        velocityThreshold=0.3,
        vis=False,
        verbose=False,
    ):
        skel = skel.clone()
        qInit = skel.getPositions() if qInit is None else qInit
        skel.setPositions(qInit)
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
        numIter = 0
        while (
            summedVelocities > velocityThreshold
            or positionDifference > cartesianErrorThreshold
        ) and numIter < numIterMax:
            for bodyNodeIndex in range(0, skel.getNumBodyNodes()):
                bodyNode = skel.getBodyNode(bodyNodeIndex)
                currentPositions[bodyNodeIndex, :] = (
                    bodyNode.getTransform().translation()
                )
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
            numIter += 1

            if verbose:
                print("Position difference: {}".format(positionDifference))
                print("Summed velocities: {}".format(summedVelocities))
                print("Iteration: {}/{}".format(numIter, numIterMax))

            if vis:
                viewer.frame()
        q = skel.getPositions()
        if vis:
            del viewer
        del world
        del node
        del skel
        return q


class DartScene(DartVisualizer):
    def __init__(
        self,
        skel,
        q,
        loadRobot=True,
        loadCell=True,
        loadBoard=True,
        skelAlpha=None,
        robotAlpha=None,
        clipBoardAlpha=None,
    ):
        super().__init__()

        self.skel = skel
        if skelAlpha is not None:
            self.skel.setAlpha(skelAlpha)
        self.addSkeleton(self.skel)
        self.skel.setPositions(q)

        urdfLoader = URDFLoader()
        # load robot
        if loadRobot:
            self.robotSkel = urdfLoader.loadPandaAndGripper()
            qRobot = self.robotSkel.getPositions()
            qRobot[[0, 1, 3, 5]] = [0, -np.pi / 4, -2 * np.pi / 3, np.pi / 2]
            # qRobot[6] = 1
            # qRobot[7] = 0
            self.robotSkel.setPositions(qRobot)
            if robotAlpha is not None:
                self.robotSkel.setAlpha(robotAlpha)
            self.addSkeleton(self.robotSkel)

        # load cell
        if loadCell:
            self.cellSkel = urdfLoader.loadCell()
            self.addSkeleton(self.cellSkel)

        # load board
        if loadBoard:
            boardSkel = urdfLoader.loadClipBoard()
            if clipBoardAlpha is not None:
                boardSkel.setAlpha(clipBoardAlpha)
            self.addSkeleton(boardSkel)

    def loadFixture(self, x=0, y=0, z=0, rx=0, ry=0, rz=0, alpha=1):
        urdfLoader = URDFLoader()
        fixtureSkel = urdfLoader.loadFixture()
        fixtureSkel.getRootJoint().setPositions(np.array([rx, ry, rz, x, y, z]))
        fixtureSkel.setAlpha(alpha)
        self.addSkeleton(fixtureSkel)
        return

    def setupSceneViewInWindow(self, x, y, width, height):
        self.viewer.setUpViewInWindow(x, y, width, height)

    def saveScene(
        self, savePath, format=".png", waitForRenderingTime=0.5, waitForSaveTime=3
    ):
        self.viewer.frame()
        self.viewer.captureScreen(savePath + format)
        self.viewer.frame()
        return

    def setModelColor(self, color=None):
        color = [0, 0, 1] if color is None else color
        bodyNodes = self.skel.getBodyNodes()
        for bodyNode in bodyNodes:
            shapeNodes = bodyNode.getShapeNodes()
            for shapeNode in shapeNodes:
                shapeNode.getVisualAspect().setColor(color)

    def setRobotPosition(self, q_robot):
        if self.robotSkel is not None:
            self.robotSkel.setPositions(q_robot)
        else:
            warn(
                "No robot loaded in Simulation yet. Cannot set robot position. First load the robot."
            )
        return
