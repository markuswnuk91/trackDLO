import os, sys
import numpy as np
import math
import dartpy as dart


class DeformableLinearObject:
    """
    Class for representing a Deformable Linear Object (DLO) in Dynamics and Animation Robotics Toolbox (DART).

    Attributes:
    skel (dart.dynamcis.Skeleton): Skeleton Representation as a Dart Skelton (refer to Dart's skeleton class doucmentation for futher details)
    length (float): Length of the DLO
    radius (float): Radius of the DLO
    density (float): Density of the DLO
    name (str): Name of the skeleton. If None name is generated automatically.
    stiffness(float): stiffness of the DLO.
    dampint(float): daming of the DLO.
    color (np.array): Color of the DLO in RGB values, e.g. [0,0,1] for blue.
    gravity (bool): If the DLO should be affected by gravity. If None defaults to true.
    collidble (bool): If the DLO is collidable. If None defaults to true.
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
        if gravity is None:
            self.gravityMode = True
        else:
            self.gravity = gravity

        if collidable is None:
            self.collidable = True
        else:
            self.collidable = collidable
        # disable adjacent body collision check by default
        self.skel.setAdjacentBodyCheck(False)
        # enable selfCollisionChecking by default
        self.skel.enableSelfCollisionCheck()

        self.makeRootBody()

        for i in range(self.numSegments - 1):
            self.addBody(self.skel.getBodyNodes()[-1])
            i += 1

        print("Succesfully created Skeleton: " + self.name)

    def makeRootBody(self):
        # rootJoint properties
        rootjoint_prop = dart.dynamics.FreeJointProperties()
        rootjoint_prop.mName = self.name + "_root" + "_joint"
        rootjoint_prop.mRestPositions = np.zeros(6)
        rootjoint_prop.mSpringStiffnesses = np.ones(6)
        rootjoint_prop.mDampingCoefficients = np.ones(6)

        # rootbody properties
        rootbody_aspect_prop = dart.dynamics.BodyNodeAspectProperties(
            name=self.name + "_root_body"
        )
        rootbody_prop = dart.dynamics.BodyNodeProperties(rootbody_aspect_prop)

        # create joint&bodyNode pair
        [rootjoint, rootbody] = self.skel.createFreeJointAndBodyNodePair(
            None, rootjoint_prop, rootbody_prop
        )
        rootbody.setGravityMode(self.gravity)
        rootbody.setCollidable(self.collidable)

        # set shapes
        self.setBodyShape_Cylinder(
            rootbody, radius=self.radius, length=self.segmentLength, color=self.color
        )

        # set the transformation between rootjoint and rootbody
        tf = dart.math.Isometry3()
        bodyNodeCenter = [0, 0, -self.segmentLength / 2.0]
        tf.set_translation(bodyNodeCenter)
        rootjoint.setTransformFromChildBodyNode(tf)

        self.setJointShape_Ball(body=rootbody, radius=self.radius)

    def addBody(self, parentNode, name=None, offset=0.0):
        joint_prop = dart.dynamics.BallJointProperties()
        if name is None:
            joint_prop.mName = (
                self.name + "_" + str(self.skel.getNumBodyNodes()) + "_joint"
            )
        else:
            joint_prop.mName = name + "_joint"

        joint_prop.mRestPositions = np.zeros(3)
        joint_prop.mSpringStiffnesses = np.ones(3) * self.stiffness
        joint_prop.mDampingCoefficients = np.ones(3) * self.damping
        joint_prop.mT_ParentBodyToJoint.set_translation(
            [0, 0, self.segmentLength / 2.0 + offset]
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
            body, radius=self.radius, length=self.segmentLength, color=self.color
        )

        self.setJointShape_Ball(body=body, radius=self.radius)

    def setJointShape_Ball(self, body, radius, color=[0, 0, 1]):
        ballShape = dart.dynamics.SphereShape(radius)
        shape_node = body.createShapeNode(ballShape)
        visual = shape_node.createVisualAspect()
        visual.setColor(color)
        tf = dart.math.Isometry3()
        jointCenter = [0, 0, -self.segmentLength / 2.0]
        shape_node.setRelativeTransform(tf)

    def setBodyShape_Cylinder(self, body, radius, length, color=[0, 0, 1]):
        cylinderShape = dart.dynamics.CylinderShape(radius, length)
        shape_node = body.createShapeNode(cylinderShape)
        visual = shape_node.createVisualAspect()
        shape_node.createCollisionAspect()
        shape_node.createDynamicsAspect()
        visual.setColor(color)


# class BDLO:
#     """
#     Class implementing a discription for BDLOs based on a graph representation to interface with dart's skeleton class.
#     A BDLO is a collection of branches.

#     Attributes:
#         name (str): name of the BDLO
#         branches (list of branches): The branches the BDLO consists of.
#         skel (dart.dynamics.Skeleton): dart skeleton used to simulate the BDLO.
#     """

#     ID = 0

#     def __init__(self, name, skel) -> None:

#         self.name = name
#         self.skel = dart.dynamics.Skeleton(name=name)

#     def addBranch(
#         self,
#         branchLength: float,
#         radius: float,
#         discretization: int,
#         startNode: node = None,
#         endNode: node = None,
#     ):
#         """
#         Branches

#         Args:
#             startNode (node): starting node the branch is connected to.
#             endNode (node): end node of the branch
#             length (float): length of the branch in m
#             radius (float): radius of the branch in m
#             discretization (int): number of segements the dart model should have for this branch.
#         """
#         segmentLength = branchLength / discretization
#         edgeWeihgts = {"length": segmentLength, "radius": radius}
#         if startNode is None:
#             nodes = []
#             rootNode = node()
#             nodes.append(rootNode)
#             for i in range(discretization):
#                 newNode = node(nodes[-1], edgeWeihgts)
#                 nodes.append(newNode)
#             self.branches.append(
#                 branch("Branch_0", rootNode, nodes[-1], branchLength, radius)
#             )
#         elif endNode is None:
#             nodes = []
#             nodes.append(startNode)
#             for i in range(discretization):
#                 newNode = node(nodes[-1], edgeWeihgts)
#                 nodes.append(newNode)
#             self.branches.append(
#                 branch(
#                     "Branch_" + str(self.branches[-1].ID),
#                     startNode,
#                     nodes[-1],
#                     branchLength,
#                     radius,
#                 )
#             )
#         else:
#             self.branches.append(
#                 branch(
#                     "Branch_" + str(self.branches[-1].ID),
#                     startNode,
#                     endNode,
#                     branchLength,
#                     radius,
#                 )
#             )
