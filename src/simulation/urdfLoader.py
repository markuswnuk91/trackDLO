import dartpy as dart
import os, sys
import numpy as np
from scipy.spatial.transform import Rotation as R
from math import pi as pi


class URDFLoader(object):
    """
    URDF Loader for table and Panda for DART
    Note that the URDF Files have to exist for this to work, copy the whole folder if it is neccesary
    """

    def __init__(self, urdfFilePath=None) -> None:
        if urdfFilePath is None:
            self.urdfFilePath = os.path.dirname(
                os.path.realpath(__file__).replace(
                    "/src/simulation", "/resources/geometries"
                )
            )
        else:
            self.urdfFilePath = urdfFilePath

    def loadPandaAndGripper(
        self, urdfFilePath=None, initialJointConfig: np.array = None
    ):
        urdfFilePath = self.urdfFilePath if urdfFilePath is None else urdfFilePath
        if initialJointConfig is None:
            initialJointConfig = np.array(
                [
                    -pi / 2,
                    -pi * 3 / 8,
                    0,
                    -pi * 5 / 8,
                    0,
                    3 / 4 * pi,
                    pi / 4,
                ]
            )
        else:
            initialJointConfig = initialJointConfig

        urdfParser = dart.utils.DartLoader()
        packageName = "package://franka_description/panda_arm_hand.urdf"
        # scriptPath = os.path.dirname(os.path.realpath(__file__))

        pandaPackagePath = os.path.join(urdfFilePath, "Panda")

        if not os.path.isdir(pandaPackagePath):
            raise NotADirectoryError(
                "Package path %s does not exist." % pandaPackagePath
            )

        # DEPRECATED since DART 7
        urdfParser.addPackageDirectory("franka_description", pandaPackagePath)
        panda = urdfParser.parseSkeleton(packageName)

        rot = R.from_euler("xyz", [0, 0, 0], degrees=True).as_matrix()

        panda.getRootJoint().setTransformFromChildBodyNode(
            dart.math.Isometry3(rot, [0, 0, -0.05])
        )

        # open gripper to avoid initial collision
        panda.setPositions([7, 8], [0.04, 0.04])
        panda.setVelocities([7, 8], [0, 0])
        panda.setForces([7, 8], [0, 0])

        # Create a Marker at the EE position
        [EEjoint, EEbody] = panda.createWeldJointAndBodyNodePair(panda.getBodyNode(7))
        iso = dart.math.Isometry3()
        iso.set_translation(np.array([0, 0, 0.209]))

        # Notiz: Rotation 45° um Z
        rot = R.from_euler("xyz", [0, 0, 45], degrees=True).as_matrix()
        iso.set_rotation(rot.T)

        EEjoint.setTransformFromParentBodyNode(iso)
        EEbody.setName("EEmiddle")

        # Set Collision for Panda False
        for i in range(panda.getNumBodyNodes()):
            body = panda.getBodyNode(i)
            body.setCollidable(False)

        if initialJointConfig.size == 7:
            panda.setPositions(np.arange(0, 7), initialJointConfig)
        else:
            print(
                "Expected 7 axes angles to set initial position for Panda. "
                "Received %s" % initialJointConfig.tostring()
            )

        return panda

    def loadCell(self, urdfFilePath=None):
        urdfFilePath = self.urdfFilePath if urdfFilePath is None else urdfFilePath

        loader = dart.utils.DartLoader()

        getRelPath = os.path.join(urdfFilePath, "Cell")

        # Raise error if Package Path to CableY is not available
        if not os.path.isdir(getRelPath):
            raise NotADirectoryError("Package path %s does not exist." % getRelPath)

        packageName = "package://Cell/cell.urdf"

        loader.addPackageDirectory("Cell", getRelPath)
        cell = loader.parseSkeleton(packageName)

        # Note that x and y direction are set according to the demonstrator setup,
        position = cell.getRootJoint().getPositions()
        # position[3:6] = [-0.435, -0.528, 0.175]
        cell.getRootJoint().setPositions(position)

        cell.getRootJoint().setActuatorType(dart.dynamics.Joint.ActuatorType.LOCKED)

        return cell

    def loadClipBoard(self, urdfFilePath=None):
        urdfFilePath = self.urdfFilePath if urdfFilePath is None else urdfFilePath
        # Load the CableY from URDF File and place it accordingly to the demonstrator
        loader = dart.utils.DartLoader()
        getRelPathCableY = os.path.join(urdfFilePath, "ClipBoard")

        # Raise error if Package Path to CableY is not available
        if not os.path.isdir(getRelPathCableY):
            raise NotADirectoryError(
                "Package path %s does not exist." % getRelPathCableY
            )

        packageName = "package://ClipBoard/clipBoard.urdf"

        loader.addPackageDirectory("ClipBoard", getRelPathCableY)
        skel = loader.parseSkeleton(packageName)

        # Note that x and y direction are set according to the demonstrator setup,
        position = skel.getRootJoint().getPositions()
        # position[3:6] = [-0.435, -0.528, 0.175]
        skel.getRootJoint().setPositions(position)

        skel.getRootJoint().setActuatorType(dart.dynamics.Joint.ActuatorType.LOCKED)

        return skel

    def loadFixture(self, urdfFilePath=None):
        urdfFilePath = self.urdfFilePath if urdfFilePath is None else urdfFilePath
        # Load the CableY from URDF File and place it accordingly to the demonstrator
        loader = dart.utils.DartLoader()
        directoryPath = os.path.join(urdfFilePath, "Fixture")

        # Raise error if Package Path to CableY is not available
        if not os.path.isdir(directoryPath):
            raise NotADirectoryError("Package path %s does not exist." % directoryPath)

        packageName = "package://Fixture/fixture.urdf"

        loader.addPackageDirectory("Fixture", directoryPath)
        skel = loader.parseSkeleton(packageName)

        # Note that x and y direction are set according to the demonstrator setup,
        position = skel.getRootJoint().getPositions()
        position[3:6] = [0, 0, 0.175]
        skel.getRootJoint().setPositions(position)

        skel.getRootJoint().setActuatorType(dart.dynamics.Joint.ActuatorType.LOCKED)

        return skel
