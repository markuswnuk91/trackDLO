import os, sys
import numpy as np
import math
import dartpy as dart

try:
    sys.path.append(os.getcwd().replace("/src/simulation", ""))
    from src.simulation.dlo import DeformableLinearObject
except:
    print("Imports for DLO failed.")
    raise


# class BranchedDeformableLinearObject(DeformableLinearObject):
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
