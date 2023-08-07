import os, sys
import numpy as np
import math
import dartpy as dart


class DeformableLinearObject(object):
    """
    Class for representing a Deformable Linear Object (DLO) in Dynamics and Animation Robotics Toolbox (DART).

    Attributes:
    skel (dart.dynamcis.Skeleton): Skeleton Representation as a Dart Skelton (refer to Dart's skeleton class doucmentation for futher details)
    frames(dict of list of dart.dynamics.SimpleFrame): SimpleFrame can be attached to the DLO. The DLO stores simple frames in a dict, where each entry contains a list of dart.dynamics.SimpleFrame.
    length (float): Length of the DLO
    radius (float): Radius of the DLO
    density (float): Density of the DLO
    name (str): Name of the skeleton. If None name is generated automatically.
    bendingStiffness(float): bendingStiffness of the DLO.
    torsinalStiffness(float): bendingStiffness of the DLO.
    bendingDampingCoeffs(float): bendingDampingCoeffs of the DLO.
    torsionalDampingCoeffs(float): torsionalDampingCoeffs of the DLO.
    color (np.array): Color of the DLO in RGB values, e.g. [0,0,1] for blue.
    gravity (bool): If the DLO should be affected by gravity. If None defaults to true.
    collidable (bool): If the DLO is collidable. If None defaults to true.
    segmentLengths (list): list of segment lenghts for each body
    frames (dict): dict of frames for frame semantics
    """

    ID = 0

    def __init__(
        self,
        numSegments=None,
        length=None,
        radius=None,
        density=None,
        name=None,
        bendingStiffness=None,
        torsionalStiffness=None,
        bendingDampingCoeffs=None,
        torsionalDampingCoeffs=None,
        color=None,
        rootJointRestPositions=None,
        gravity=True,
        collidable=None,
        adjacentBodyCheck=None,
        enableSelfCollisionCheck: bool = None,
        verbose=None,
        *args,
        **kwargs,
    ):
        self.ID = DeformableLinearObject.ID
        DeformableLinearObject.ID += 1
        if name is None:
            self.name = "DLO_" + str(self.ID)
            self.skel = dart.dynamics.Skeleton(name="DLO_" + str(self.ID))
        else:
            self.name = name
            self.skel = dart.dynamics.Skeleton(name=self.name)

        self.numSegments = 20 if numSegments is None else numSegments
        self.length = 1 if length is None else length
        self.radius = 0.01 if radius is None else radius
        self.density = 1000 if density is None else density
        self.bendingStiffness = 1 if bendingStiffness is None else bendingStiffness
        self.torsionalStiffness = (
            1 if torsionalStiffness is None else torsionalStiffness
        )
        self.bendingDampingCoeffs = (
            0.1 if bendingDampingCoeffs is None else bendingDampingCoeffs
        )
        self.torsionalDampingCoeffs = (
            0.1 if torsionalDampingCoeffs is None else torsionalDampingCoeffs
        )
        self.color = np.array([0, 0, 1]) if color is None else color
        self.rootJointRestPositions = (
            np.zeros(6) if rootJointRestPositions is None else rootJointRestPositions
        )
        self.gravity = False if gravity is None else gravity
        self.collidable = False if collidable is None else collidable
        self.adjacentBodyCheck = (
            False if adjacentBodyCheck is None else adjacentBodyCheck
        )
        self.enableSelfCollisionCheck = (
            False if enableSelfCollisionCheck is None else enableSelfCollisionCheck
        )
        self.verbose = False if verbose is None else verbose

        self.segmentLength = self.length / self.numSegments
        self.segmentLengths = []
        self.frames = {}

        self.makeRootBody(
            segmentLength=self.segmentLength,
            radius=self.radius,
            density=self.density,
            color=self.color,
            restPositions=self.rootJointRestPositions,
        )

        for i in range(self.numSegments - 1):
            self.addBody(
                parentNode=self.skel.getBodyNodes()[-1],
                segmentLength=self.segmentLength,
                radius=self.radius,
                density=self.density,
                stiffnesses=np.array(
                    [
                        self.bendingStiffness,
                        self.bendingStiffness,
                        self.torsionalStiffness,
                    ]
                ),
                dampingCoeffs=np.array(
                    [
                        self.bendingDampingCoeffs,
                        self.bendingDampingCoeffs,
                        self.torsionalDampingCoeffs,
                    ]
                ),
                restPositions=np.zeros(3),
                color=self.color,
            )
            i += 1

        # disable adjacent body collision check by default
        self.skel.setAdjacentBodyCheck(self.adjacentBodyCheck)

        # enable selfCollisionChecking by default
        # if self.enableSelfCollisionCheck:
        #     self.skel.enableSelfCollisionCheck()
        # else:
        #     self.skel.disableSelfCollisionCheck()
        self.skel.setSelfCollisionCheck(self.enableSelfCollisionCheck)
        if self.verbose:
            print("Successfully created Skeleton: " + self.name)

    def makeRootBody(
        self,
        segmentLength: float,
        radius: float,
        density: float,
        stiffnesses=None,
        dampingCoeffs=None,
        restPositions=None,
        color=None,
        name=None,
    ):
        restPositions = np.zeros(6) if restPositions is None else restPositions
        stiffnesses = (
            np.zeros(len(restPositions)) if stiffnesses is None else stiffnesses
        )
        dampingCoeffs = (
            np.zeros(len(restPositions)) if dampingCoeffs is None else dampingCoeffs
        )
        color = np.array([0, 0, 1]) if color is None else color

        # rootJoint properties
        if len(restPositions) == 6:
            rootjoint_prop = dart.dynamics.FreeJointProperties()
        elif len(restPositions) == 3:
            rootjoint_prop = dart.dynamics.BallJointProperties()
        else:
            raise ValueError(
                "Only free- (6DoF) and ball (3Dof) joint types are currently supported. Got {} Dofs for the rest positions".format(
                    len(restPositions)
                )
            )

        if name is None:
            rootjoint_prop.mName = self.name + "_root" + "_joint"
        else:
            rootjoint_prop.mName = name + "_root" + "_joint"
        rootjoint_prop.mRestPositions = restPositions
        rootjoint_prop.mInitialPositions = restPositions
        rootjoint_prop.mSpringStiffnesses = stiffnesses
        rootjoint_prop.mDampingCoefficients = dampingCoeffs

        # rootbody properties
        if name is None:
            rootbody_aspect_prop = dart.dynamics.BodyNodeAspectProperties(
                name=self.name + "_root_body"
            )
        else:
            rootbody_aspect_prop = dart.dynamics.BodyNodeAspectProperties(name=name)

        rootbody_prop = dart.dynamics.BodyNodeProperties(rootbody_aspect_prop)

        # create joint&bodyNode pair
        if len(rootjoint_prop.mInitialPositions) == 6:
            [rootjoint, rootbody] = self.skel.createFreeJointAndBodyNodePair(
                None, rootjoint_prop, rootbody_prop
            )
        elif len(rootjoint_prop.mInitialPositions) == 3:
            [rootjoint, rootbody] = self.skel.createBallJointAndBodyNodePair(
                None, rootjoint_prop, rootbody_prop
            )

        rootbody.setGravityMode(self.gravity)
        rootbody.setCollidable(self.collidable)
        self.segmentLengths.append(segmentLength)

        # set shapes
        if (segmentLength - 2 * radius) <= 0:
            bodyLength = segmentLength
        else:
            bodyLength = segmentLength - 2 * radius
        self.setBodyShape_Cylinder(
            rootbody,
            radius=radius,
            length=bodyLength,
            color=color,
            density=density,
        )

        # set the transformation between rootjoint and rootbody
        tf = dart.math.Isometry3()
        bodyNodeCenter = [0, 0, -segmentLength / 2.0]
        tf.set_translation(bodyNodeCenter)
        rootjoint.setTransformFromChildBodyNode(tf)

        self.setJointShape_Ball(body=rootbody, radius=radius, color=color)

    def addBody(
        self,
        parentNode,
        segmentLength: float,
        radius: float,
        density: float,
        stiffnesses: np.array,
        dampingCoeffs: np.array,
        restPositions: np.array = np.zeros(3),
        color: np.array = np.array([0, 0, 1]),
        offset: float = 0.0,
        name: str = None,
    ):
        joint_prop = dart.dynamics.BallJointProperties()
        if name is None:
            joint_prop.mName = (
                self.name + "_" + str(self.skel.getNumBodyNodes()) + "_joint"
            )
        else:
            joint_prop.mName = name + "_joint"

        joint_prop.mRestPositions = restPositions
        joint_prop.mInitialPositions = restPositions
        joint_prop.mSpringStiffnesses = stiffnesses
        joint_prop.mDampingCoefficients = dampingCoeffs
        joint_prop.mT_ParentBodyToJoint.set_translation(
            [0, 0, segmentLength / 2.0 + offset]
        )
        if name is None:
            body_aspect_prop = dart.dynamics.BodyNodeAspectProperties(
                name=(self.name + "_" + str(self.skel.getNumBodyNodes()) + "_body")
            )
        else:
            body_aspect_prop = dart.dynamics.BodyNodeAspectProperties(name=name)

        body_prop = dart.dynamics.BodyNodeProperties(body_aspect_prop)

        # add the body
        [joint, body] = self.skel.createBallJointAndBodyNodePair(
            parentNode, joint_prop, body_prop
        )
        body.setGravityMode(self.gravity)
        body.setCollidable(self.collidable)
        self.segmentLengths.append(segmentLength)

        if (segmentLength - 2 * radius) <= 0:
            bodyLength = segmentLength
        else:
            bodyLength = segmentLength - 2 * radius
        self.setBodyShape_Cylinder(
            body,
            radius=radius,
            length=bodyLength,
            color=color,
            density=density,
        )
        # set the transformation between parent joint and body
        tf = dart.math.Isometry3()
        bodyNodeCenter = [0, 0, -segmentLength / 2.0]
        tf.set_translation(bodyNodeCenter)
        joint.setTransformFromChildBodyNode(tf)

        self.setJointShape_Ball(body=body, radius=radius, color=color)

    def setJointShape_Ball(self, body, radius, color=[0, 0, 1]):
        ballShape = dart.dynamics.SphereShape(radius)
        shape_node = body.createShapeNode(ballShape)
        visual = shape_node.createVisualAspect()
        visual.setColor(color)
        tf = dart.math.Isometry3()
        jointCenter = (
            body.getParentJoint().getTransformFromChildBodyNode().translation()
        )
        tf.set_translation(jointCenter)
        shape_node.setRelativeTransform(tf)

    def setBodyShape_Cylinder(self, body, radius, length, density, color=[0, 0, 1]):
        cylinderShape = dart.dynamics.CylinderShape(radius, length)
        shape_node = body.createShapeNode(cylinderShape)
        visual = shape_node.createVisualAspect()
        shape_node.createCollisionAspect()
        shape_node.createDynamicsAspect()
        visual.setColor(color)

        # set mass and inertia of the body
        inertia = cylinderShape.computeInertiaFromDensity(density)
        volume = cylinderShape.getVolume()
        inertia_dart = dart.dynamics.Inertia()
        inertia_dart.setMoment(inertia)
        body.setInertia(inertia_dart)
        body.setMass(volume * density)

    def getNumSegments(self):
        return self.skel.getNumBodyNodes()

    def setInitialPosition(self, dof: int, q: float):
        """
        Set the initial position for a degree of freedom of the DLO.
        Also sets the rest position for this degree of freedom to the given intial position.

        Args:
            dof (int): degree of freedom for which the initial position should be set
            q (float): initial position for this degree of freedom.
        """
        self.skel.getDof(dof).setPosition(q)
        self.skel.getDof(dof).setRestPosition(self.skel.getDof(dof).getPosition())

    def getRestPosition(self, dof: int, q: float):
        return self.skel.getDof(dof).getRestPosition()

    def getPosition(self, dof: int):
        return self.skel.getDof(dof).getPosition()

    def getPositions(self):
        return self.skel.getPositions()

    def setPositions(self, q):
        self.skel.setPositions(q)

    def addSimpleFrame(
        self,
        key: str,
        bodyNodeNumber: int,
        relTransform: np.ndarray = np.eye(4),
        name: str = None,
        shape: dart.dynamics.Shape = None,
        shapeColor: np.array = np.array([0, 0, 1]),
    ):
        """
        Adds a simple frame to this DLO. Simple frames are grouped by keywords.
        For each key the DLO holds a list of dart.dynamics.SimpleFrames

        Args:
            key (str): The key specifying the group for the simple frame to add, e.g.
            "Marker" or "Target".
            bodyNodeNumber (int): The index number of the body node the simple frame should be attached to.
            relTransform (dart.math.Isometry3): the relativeTransformation in the local coordinate system of the body node.
        """
        # check if key already exists
        if key in self.frames:
            frameList = self.frames[key]
        else:
            frameList = []

        if name is None:
            frameName = self.name + "_" + key + "_" + str(len(frameList))
        else:
            frameName = name
        tf = dart.math.Isometry3()
        tf.set_translation(relTransform[:3, 3])
        tf.set_rotation(relTransform[:3, :3])
        newFrame = dart.dynamics.SimpleFrame(
            self.skel.getBodyNode(bodyNodeNumber), frameName, relTransform
        )

        # set the shape if given
        if shape is not None:
            newFrame.setShape(shape)
            visualAspect = newFrame.createVisualAspect()
            visualAspect.setColor(shapeColor)

        frameList.append(newFrame)
        self.frames[key] = frameList
