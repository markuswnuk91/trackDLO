import os, sys
import numpy as np
import math
import dartpy as dart

try:
    sys.path.append(os.getcwd().replace("/src/simulation", ""))
    from src.simulation.dlo import DeformableLinearObject
    from src.simulation.topologyTree import topologyTree
except:
    print("Imports for DLO failed.")
    raise


class bdloDiscretization(topologyTree):
    """
    A bdloTopology is a topologyTree with additional information about the discretization, specifying the desired number of segments which should be used in the simulaton to represent each branch.

    The information for the discretization is stored as a list of tupels containing each branch with its corresponding discretization.

    Attributes:
        see topologyTree
        discretization (list of tuple): Tuples of branches together with the desired number of segments
    """

    def __init__(self):
        super().__init__()

        # add additional information about the discretization
        # self.discretization = ...


class BranchedDeformableLinearObject(DeformableLinearObject):
    """
    Class implementing a interface for handling Branched Defromable Linear Objects (BDLO) with dart's skeleton class.
    The class consists of a topologyTree descring its topology and a dartSkeleton used for simulation.

    Attributes:
        name (str): name of the BDLO
    """

    ID = 0

    def __init__(self, name, topology: topologyTree) -> None:

        self.ID = BranchedDeformableLinearObject.ID
        BranchedDeformableLinearObject.ID += 1

        if name is None:
            self.name = "BDLO_" + str(self.ID)
        else:
            self.name = name

        self.topology = topology
        self.skel = dart.dynamics.Skeleton(name=self.name)

        for i, branch in enumerate(self.topology.getBranches()):
            branchLength = branch.getBranchInfo["branchLength"]
            branchRadius = branch.getBranchInfo["branchRadius"]
            numSegments = branch.getBranchInfo["numSegments"]
            branchColor = branch.getBranchInfo["color"]

            if i == 0 and branch.getStartNode().getParentNode() is not None:
                raise ValueError(
                    "Expected the first branch to start with the RootNode, but got branch with startNode that has parent: ".format(
                        branch.getStartNode().getParentNode()
                    )
                )
            elif i == 0 and branch.getStartNode().getParentNode() is None:
                self.makeRootBody(
                    segmentLength=branchLength / numSegments,
                    radius=branchRadius,
                    color=branchColor,
                )
                # branch.addBranchInfo["dartIndices"] = [self.skel.getNumBodyNodes()]

                # generate the rest of the rootBranch here

                # determine the branchNodes belongign to the branch
                # determine the leafNodes belonging to the branch
                # set the corresponding dartIndices as nodeInfo

            else:
                # determine the branchNodes belongign to the branch

                # if both have already dart indices something went wrong

                # determine the parentBodyNode in the dart skeleton
                # parentBodyNode =
                self.addBody()
                dartIndices = branch.getBranchInfo["dartIndices"]
                dartIndices.append(self.skel.getNumBodyNodes())
                branch.setBranchInfo["dartIndices"] = dartIndices

    def getBranchBodyNodes(self, branchNumber):
        """
        Returns the dart bodyNodes corresponding to a branch
        """

    def getBranchBodyNodeIndices(self, branchNumber):
        """
        Returns the dart bodyNodes indices corresponding to a branch
        """

    def getLeafNodeBodyNodes(self):
        """
        Returns the DART bodyNodes corresponding to the leafNodes in the topology of the BDLO
        """

    def getBranchNodeBodyNodes(self):
        """
        Returns the DART bodyNodes corresponding to the branchNodes in the topology of the BDLO
        """
