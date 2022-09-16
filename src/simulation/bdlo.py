import numpy as np
import dartpy as dart


class DeformableLinearObject:
    """
    Class for representing a Deformable Linear Object (DLO) in Dynamics and Animation Robotics Toolbox (DART).

    Attributes:
    skel (dart.dynamcis.Skeleton): Skeleton Representation as a Dart Skelton (refer to Dart's skeleton class doucmentation for futher details)
    length (float): Length of the DLO
    radius (float): Radius of the DLO
    density (float): Density of the DLO
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

        # disable adjacent body collision check by default
        self.skel.setAdjacentBodyCheck(False)
        # enable selfCollisionChecking by default
        self.skel.enableSelfCollisionCheck()

        # rootJoint properties
        rootjoint_prop = dart.dynamics.FreeJointProperties()
        rootjoint_prop.mName = self.name + "_root" + "_joint"
        rootjoint_prop.mRestPositions = np.zeros(6)
        rootjoint_prop.mSpringStiffnesses = np.ones(6)
        rootjoint_prop.mDampingCoefficients = np.ones(6)

        # rootbody properties
        rootbody_aspect_prop = dart.dynamics.BodyNodeAspectProperties(
            name=self.name + "root_body"
        )
        rootbody_prop = dart.dynamics.BodyNodeProperties(rootbody_aspect_prop)

        # create joint&bodyNode pair
        [rootjoint, rootbody] = self.skel.createFreeJointAndBodyNodePair(
            None, rootjoint_prop, rootbody_prop
        )
        rootbody.setGravityMode(True)
        rootbody.setCollidable(True)

        # set shapes
        self.setBodyShape_Cylinder(rootbody)

        # set the transformation between rootjoint and rootbody
        tf = dart.math.Isometry3()
        bodyNodeCenter = [0, 0, self.segmentLength / 2.0]
        tf.set_translation(bodyNodeCenter)
        rootjoint.setTransformFromChildBodyNode(-tf)

    def addBody(self, parentNode, name, offset):
        joint_prop = dart.dynamics.BallJointProperties()
        joint_prop.mName = self.name + "_" + str(self.skel.getNumBodyNodes()) + "_joint"
        joint_prop.mRestPositions = np.zeros(3)
        joint_prop.mSpringStiffnesses = np.ones(3) * self.stiffness
        joint_prop.mDampingCoefficients = np.ones(3) * self.damping
        joint_prop.mT_ParentBodyToJoint.set_translation(
            [0, 0, self.segmentLength + offset]
        )
        body_aspect_prop = dart.dynamics.BodyNodeAspectProperties(name)
        body_prop = dart.dynamics.BodyNodeProperties(body_aspect_prop)

    def setJointShape_Ball(self, joint, radius, color=[0, 0, 1]):
        ballShape = dart.dynamics.BallShape(radius)

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
