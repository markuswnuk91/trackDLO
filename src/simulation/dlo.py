import os, sys
import numpy as np
import math
import dartpy as dart


class DeformableLinearObject:
    """
    Class for representing a Deformable Linear Object (DLO) in Dynamics and Animation Robotics Toolbox (DART).

    Attributes:
    skel (dart.dynamcis.Skeleton): Skeleton Representation as a Dart Skelton (refer to Dart's skeleton class doucmentation for futher details)
    frames(dict of list of dart.dynamics.SimpleFrame): SimpleFrame can be attached to the DLO. The DLO stores simple frames in a dict, where each entry contains a list of dart.dynamics.SimpleFrame.
    length (float): Length of the DLO
    radius (float): Radius of the DLO
    density (float): Density of the DLO
    name (str): Name of the skeleton. If None name is generated automatically.
    stiffness(float): stiffness of the DLO.
    dampint(float): daming of the DLO.
    color (np.array): Color of the DLO in RGB values, e.g. [0,0,1] for blue.
    gravity (bool): If the DLO should be affected by gravity. If None defaults to true.
    collidable (bool): If the DLO is collidable. If None defaults to true.
    """

    ID = 0

    def __init__(
        self,
        numSegments: int = 20,
        length: float = 1,
        radius: float = 0.01,
        density: float = 1000,
        name: str = None,
        stiffness: float = 1,
        damping: float = 0.1,
        color: np.array = np.array([0, 0, 1]),
        gravity: bool = True,
        collidable: bool = True,
        adjacentBodyCheck: bool = False,
        enableSelfCollisionCheck: bool = True,
    ):

        self.ID = DeformableLinearObject.ID
        DeformableLinearObject.ID += 1
        if name is None:
            self.name = "DLO_" + str(self.ID)
            self.skel = dart.dynamics.Skeleton(name="DLO_" + str(self.ID))
        else:
            self.name = name
            self.skel = dart.dynamics.Skeleton(name=self.name)

        self.numSegments = numSegments
        self.length = length
        self.radius = radius
        self.density = density
        self.stiffness = stiffness
        self.damping = damping
        self.color = color
        self.segmentLength = self.length / self.numSegments
        self.adjacentBodyCheck = adjacentBodyCheck
        self.enableSelfCollisionCheck = enableSelfCollisionCheck
        self.frames = {}

        if gravity is None:
            self.gravity = True
        else:
            self.gravity = gravity

        if collidable is None:
            self.collidable = True
        else:
            self.collidable = collidable

        self.makeRootBody(
            segmentLength=self.segmentLength,
            radius=self.radius,
            density=self.density,
            color=self.color,
        )

        for i in range(self.numSegments - 1):
            self.addBody(
                parentNode=self.skel.getBodyNodes()[-1],
                segmentLength=self.segmentLength,
                radius=self.radius,
                density=self.density,
                stiffnesses=np.ones(3) * self.stiffness,
                dampingCoeffs=np.ones(3) * self.damping,
                restPositions=np.zeros(3),
                color=np.array([0, 0, 1]),
            )
            i += 1

        # disable adjacent body collision check by default
        self.skel.setAdjacentBodyCheck(self.adjacentBodyCheck)

        # enable selfCollisionChecking by default
        if self.enableSelfCollisionCheck:
            self.skel.enableSelfCollisionCheck()
        else:
            self.skel.disableSelfCollisionCheck()

        print("Succesfully created Skeleton: " + self.name)

    def makeRootBody(
        self,
        segmentLength: float,
        radius: float,
        density: float,
        color: np.array = np.array([0, 0, 1]),
        name: str = None,
    ):

        # rootJoint properties
        rootjoint_prop = dart.dynamics.FreeJointProperties()
        if name is None:
            rootjoint_prop.mName = self.name + "_root" + "_joint"
        else:
            rootjoint_prop.mName = name + "_root" + "_joint"
        rootjoint_prop.mRestPositions = np.zeros(6)
        rootjoint_prop.mSpringStiffnesses = np.zeros(6)
        rootjoint_prop.mDampingCoefficients = np.zeros(6)

        # rootbody properties
        if name is None:
            rootbody_aspect_prop = dart.dynamics.BodyNodeAspectProperties(
                name=self.name + "_root_body"
            )
        else:
            rootbody_aspect_prop = dart.dynamics.BodyNodeAspectProperties(name=name)

        rootbody_prop = dart.dynamics.BodyNodeProperties(rootbody_aspect_prop)

        # create joint&bodyNode pair
        [rootjoint, rootbody] = self.skel.createFreeJointAndBodyNodePair(
            None, rootjoint_prop, rootbody_prop
        )
        rootbody.setGravityMode(self.gravity)
        rootbody.setCollidable(self.collidable)

        # set shapes
        self.setBodyShape_Cylinder(
            rootbody,
            radius=radius,
            length=(segmentLength - 2 * radius),
            color=color,
            density=density,
        )

        # set the transformation between rootjoint and rootbody
        tf = dart.math.Isometry3()
        bodyNodeCenter = [0, 0, -segmentLength / 2.0]
        tf.set_translation(bodyNodeCenter)
        rootjoint.setTransformFromChildBodyNode(tf)

        self.setJointShape_Ball(body=rootbody, radius=radius)

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
        [joint, body] = self.skel.createBallJointAndBodyNodePair(
            parentNode, joint_prop, body_prop
        )
        body.setGravityMode(self.gravity)
        body.setCollidable(self.collidable)

        self.setBodyShape_Cylinder(
            body,
            radius=radius,
            length=(segmentLength - 2 * radius),
            color=color,
            density=density,
        )
        # set the transformation between parent joint and body
        tf = dart.math.Isometry3()
        bodyNodeCenter = [0, 0, -segmentLength / 2.0]
        tf.set_translation(bodyNodeCenter)
        joint.setTransformFromChildBodyNode(tf)

        self.setJointShape_Ball(body=body, radius=radius)

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