import sys, os
import numpy as np
import dartpy as dart
import time

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

    def addPointCloud(self, points, colors=None):
        """
        Visualize a PointCloud in DART
        """
        colors = [0, 0, 0] if colors is None else colors
        pointCloudShape = dart.dynamics.PointCloudShape(0.005)
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

        self.world.addSimpleFrame(frame)

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
    def __init__(self, skel, q, loadRobot=True, loadCell=True, loadBoard=True):
        super().__init__()

        self.skel = skel
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
            self.addSkeleton(self.robotSkel)

        # load cell
        if loadCell:
            self.cellSkel = urdfLoader.loadCell()
            self.addSkeleton(self.cellSkel)

        # load board
        if loadBoard:
            boardSkel = urdfLoader.loadClipBoard()
            self.addSkeleton(boardSkel)

    def saveScene(self, savePath):
        self.showFrame()
        dir = os.path.dirname(savePath)
        file_name = os.path.basename(savePath)
        self.viewer.record(dir, file_name)

    def setModelColor(self, color=None):
        color = [0, 0, 1] if color is None else color
        bodyNodes = self.skel.getBodyNodes()
        for bodyNode in bodyNodes:
            shapeNodes = bodyNode.getShapeNodes()
            for shapeNode in shapeNodes:
                shapeNode.getVisualAspect().setColor(color)
