import os, sys
import numpy as np
import math
import dartpy as dart
from warnings import warn

try:
    sys.path.append(os.getcwd().replace("/src/simulation", ""))
    from src.simulation.dlo import DeformableLinearObject
    from src.simulation.topologyTree import topologyTree
except:
    print("Imports for DLO failed.")
    raise


class bdloSpecification(topologyTree):
    """
    A bdloSpecification is topological description of a BDLO which contains information about the branches of the BDLO which are requried to build a kinematic model.
    The class is drived from a topologicalTree but requires additional information to be stored for each branch.
    The additional information is stored as dict in the branchInfo of each branch.

    Attributes:
    branchSpecs (list of dict): branch specifications for each branch of the BDLO as a dict of parameters. For the required parameters see: branchSpec
    branchSpec (dict) with parameters as keywords:
        - length (float) : length of the branch [m]
        - radius (float): radius of the branch [m]
        - numSegments (int): desired number of segments the branch should be discretized into
        - color (np.array): color of the branch [RGB Values]
        - bendingStiffness (float): bending stiffness of the branch [N/rad]
        - torsionalStiffness (float): torsional stiffness of the branch [N/rad]
        - bendingDampingCoeffs (float): bending damping coefficients of the branch [Ns/rad]
        - torsionalDampingCoeffs (float): torsional damping coefficients of the branch[Ns/rad]
        - restPosition (np.array): rest position of the branch in [Rx, Ry, Rz] as angles of the ballJoint in bodyNode coodinates at the branch point [rad]
    """

    def __init__(
        self,
        csgraph: np.array,
        branchSpecs: list = None,
        specInfo: dict = None,
        name: str = None,
        defaultRadius=0.01,
        defaultDensity=1000,
        defaultColor=[0, 0, 1],
        defaultBendingStiffness=1,
        defaultTorsionalStiffness=1,
        defaultBendingDampingCoeff=0.1,
        defaultTorsionalDampingCoeff=0.1,
    ):
        super().__init__(csgraph, specInfo, name)

        if branchSpecs is None:
            warn("No branch specifications provided. Using default values.")
            self.branchSpecs = [{}] * len(self.branches)
        else:
            self.branchSpecs = branchSpecs

        # make sure specification contains all necessary information
        for i, branchSpec in enumerate(self.branchSpecs):
            if "length" not in branchSpec:
                warn(
                    "Expected the branch length to be specified in the branch specification, but specification has no parameter length for branch {}. Calculating length from adjacency matrix.".format(
                        i
                    )
                )
                newSpec = self.branchSpecs[i].copy()
                newSpec["length"] = self.branches[i].getBranchInfo()["length"]
                self.branchSpecs[i] = newSpec

            if "radius" not in branchSpec:
                warn(
                    "Expected the branch radius to be specified in the branch specification, but specification has no parameter radius for branch {}. Assuming default value for radius.".format(
                        i
                    )
                )
                newSpec = self.branchSpecs[i].copy()
                newSpec["radius"] = defaultRadius
                self.branchSpecs[i] = newSpec

            if "density" not in branchSpec:
                warn(
                    "Expected the branch radius to be specified in the branch specification, but specification has no parameter radius for branch {}. Assuming default value for density.".format(
                        i
                    )
                )
                newSpec = self.branchSpecs[i].copy()
                newSpec["density"] = defaultDensity
                self.branchSpecs[i] = newSpec

            if "numSegments" not in branchSpec:
                warn(
                    "Expected the branch radius to be specified in the branch specification, but specification has no parameter numSegments for branch {}.".format(
                        i
                    )
                )
                newSpec = self.branchSpecs[i].copy()
                newSpec["numSegments"] = int(
                    math.ceil(
                        self.branches[i].getBranchInfo()["length"]
                        / self.getSummedLength()
                        * 100
                    )
                )
                self.branchSpecs[i] = newSpec

            if "color" not in branchSpec:
                warn(
                    "No color information given for branch {} using default color blue ([0,0,1]).".format(
                        i
                    )
                )
                newSpec = self.branchSpecs[i].copy()
                newSpec["color"] = defaultColor
                self.branchSpecs[i] = newSpec

            if "bendingStiffness" not in branchSpec:
                warn(
                    "No bending stiffness information given for branch {} using default stiffness (1 N/rad).".format(
                        i
                    )
                )
                newSpec = self.branchSpecs[i].copy()
                newSpec["bendingStiffness"] = defaultBendingStiffness
                self.branchSpecs[i] = newSpec

            if "torsionalStiffness" not in branchSpec:
                warn(
                    "No torsional stiffness information given for branch {} using default stiffness (1 N/rad).".format(
                        i
                    )
                )
                newSpec = self.branchSpecs[i].copy()
                newSpec["torsionalStiffness"] = defaultTorsionalStiffness
                self.branchSpecs[i] = newSpec

            if "bendingDampingCoeffs" not in branchSpec:
                warn(
                    "No bending damping coefficient information given for branch {} using default stiffness (0.1 N/rad).".format(
                        i
                    )
                )
                newSpec = self.branchSpecs[i].copy()
                newSpec["bendingDampingCoeffs"] = defaultBendingDampingCoeff
                self.branchSpecs[i] = newSpec

            if "torsionalDampingCoeffs" not in branchSpec:
                warn(
                    "No torsional damping coefficient information given for branch {} using default stiffness (0.1 N/rad).".format(
                        i
                    )
                )
                newSpec = self.branchSpecs[i].copy()
                newSpec["torsionalDampingCoeffs"] = defaultTorsionalDampingCoeff
                self.branchSpecs[i] = newSpec

            if "restPosition" not in branchSpec:
                warn(
                    "No restPosition information given for branch {} using default rest position.".format(
                        i
                    )
                )
                newSpec = self.branchSpecs[i].copy()
                # branchNode
                if self.isBranchNode(self.branches[i].getStartNode()):
                    node = self.getBranchNodeFromNode(self.branches[i].getStartNode())
                    numBranches = node.getNumBranches()
                    siblingBranches = self.getSiblingBranches(self.branches[i])
                    childBranchIndices = self.getBranchIndices(siblingBranches)
                    k = childBranchIndices.index(i)
                    initRestAngle = -60 / 180 * math.pi
                    deltaAngle = 120 / 180 * math.pi / numBranches
                    restAngle = initRestAngle + k * deltaAngle

                if i == 0:
                    newSpec["restPosition"] = np.array([restAngle, 0, 0, 0, 0, 0])
                else:
                    newSpec["restPosition"] = np.array([restAngle, 0, 0])

                self.branchSpecs[i] = newSpec

        # set the branchInfo according to the specification
        for i, branch in enumerate(self.branches):
            branchInfo = branch.getBranchInfo()
            for key in self.branchSpecs[i]:
                if key in branchInfo:
                    # check if branchInfo differs from branchSpec
                    if branchInfo[key] != self.branchSpecs[i][key]:
                        raise ValueError(
                            "Ambigous information for parameter : {}. Obtained {} from adjacency matrix, and {} from branch specification. Using branch specification value in the following".format(
                                key, branchInfo[key], self.branchSpecs[i][key]
                            )
                        )
                    else:
                        pass
                else:
                    branchInfo[key] = self.branchSpecs[i][key]

    def getBranchSpecs(self):
        return self.branchSpecs

    def getBranchSpec(self, num: int):
        return self.branchSpecs[num]


class BranchedDeformableLinearObject(DeformableLinearObject):
    """
    Class implementing a interface for handling Branched Defromable Linear Objects (BDLO) with dart's skeleton class.
    The class consists of a topologyTree descring its topology and a dartSkeleton which can be used for simulation.

    Attributes:
        name (str): name of the BDLO
    """

    ID = 0

    def __init__(
        self,
        topology: bdloSpecification,
        name=None,
        gravity: bool = True,
        collidable: bool = True,
        adjacentBodyCheck: bool = False,
        enableSelfCollisionCheck: bool = True,
    ):

        self.ID = BranchedDeformableLinearObject.ID
        BranchedDeformableLinearObject.ID += 1

        if name is None:
            self.name = "BDLO_" + str(self.ID)
        else:
            self.name = name

        self.topology = topology
        self.skel = dart.dynamics.Skeleton(name=self.name)

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

        # create dartSkeleton
        unvisitedBranches = self.topology.getBranches().copy()
        blacklist = (
            []
        )  # branches for witch bodyNodes were already generated are blacklisted
        nextBranchCandidates = []
        nextBranchCandidates.append(self.topology.getBranch(0))
        while len(unvisitedBranches) > 0 and len(nextBranchCandidates) > 0:
            # get necessary information for generating the bodyNodes for the branch
            branch = nextBranchCandidates.pop(0)
            branchInfo = branch.getBranchInfo()
            length = branchInfo["length"]
            radius = branchInfo["radius"]
            density = branchInfo["density"]
            numSegments = branchInfo["numSegments"]
            color = branchInfo["color"]
            stiffness = np.array(
                [
                    branchInfo["bendingStiffness"],
                    branchInfo["bendingStiffness"],
                    branchInfo["torsionalStiffness"],
                ]
            )
            damping = np.array(
                [
                    branchInfo["bendingDampingCoeffs"],
                    branchInfo["bendingDampingCoeffs"],
                    branchInfo["torsionalDampingCoeffs"],
                ]
            )
            restPosition = branchInfo["restPosition"]
            segmentLength = length / numSegments
            correspondingBodyNodes = []

            # make sure we start at the rootNode
            if (
                branch == self.topology.getBranch(0)
                and branch.getStartNode().getParent() is not None
            ):
                raise ValueError(
                    "Expected the first branch to start with the RootNode, but got branch with startNode that has parent: ".format(
                        branch.getStartNode().getParentNode()
                    )
                )
            elif (
                branch == self.topology.getBranch(0)
                and branch.getStartNode().getParent() is None
            ):
                # generate the rootBranch
                self.makeRootBody(
                    segmentLength=segmentLength,
                    radius=radius,
                    density=density,
                    color=color,
                )
                correspondingBodyNodes.append(self.skel.getNumBodyNodes())
                for i in range(numSegments - 1):
                    self.addBody(
                        parentNode=self.skel.getBodyNodes()[-1],
                        segmentLength=segmentLength,
                        radius=radius,
                        density=density,
                        stiffnesses=np.ones(3) * stiffness,
                        dampingCoeffs=np.ones(3) * damping,
                        restPositions=np.zeros(3),
                        color=color,
                    )
                    correspondingBodyNodes.append(self.skel.getNumBodyNodes())
                    i += 1

                # add information of the corresponding bodyNodes to the branch
                branchInfo["bodyNodeIndices"] = correspondingBodyNodes
                branch.setBranchInfo(branchInfo)

                # add information of the corresponding bodyNode to the startNode
                startNode = branch.getStartNode()
                startNode.setNodeInfo({"bodyNodeIndex": correspondingBodyNodes[0]})
                if self.topology.isLeafNode(startNode) == True:
                    leafNode = self.topology.getLeafNode(startNode)
                    leafNode.setLeafNodeInfo(
                        {"bodyNodeIndex": correspondingBodyNodes[0]}
                    )
                elif self.topology.isBranchNode(startNode) == True:
                    branchNode = self.topology.getBranchNodeFromNode(startNode)
                    branchNode.setBranchNodeInfo(
                        {"bodyNodeIndex": correspondingBodyNodes[0]}
                    )

                # add information of the corresponding bodyNode to the endNode
                endNode = branch.getEndNode()
                endNode.setNodeInfo({"bodyNodeIndex": correspondingBodyNodes[-1]})
                if self.topology.isLeafNode(endNode) == True:
                    leafNode = self.topology.getLeafNodeFromNode(endNode)
                    leafNode.setLeafNodeInfo(
                        {"bodyNodeIndex": correspondingBodyNodes[-1]}
                    )
                elif self.topology.isBranchNode(endNode) == True:
                    branchNode = self.topology.getBranchNodeFromNode(endNode)
                    branchNode.setBranchNodeInfo(
                        {"bodyNodeIndex": correspondingBodyNodes[-1]}
                    )

                # get next branch candidates for which bodyNodes are generated
                childBranches = self.topology.getChildBranches(branch)
                if len(childBranches) > 0:
                    for childBranch in self.topology.getChildBranches(branch):
                        nextBranchCandidates.append(childBranch)
                unvisitedBranches.remove(branch)
                blacklist.append(branch)
            else:
                # generate bodyNodes for remaining branches
                parentBodyNodeIdx = branch.getStartNode().getNodeInfo()["bodyNodeIndex"]
                parentBodyNode = self.skel.getBodyNode(parentBodyNodeIdx)
                for i in range(numSegments):
                    self.addBody(
                        parentNode=parentBodyNode,
                        segmentLength=segmentLength,
                        radius=radius,
                        stiffnesses=np.ones(3) * stiffness,
                        dampingCoeffs=np.ones(3) * damping,
                        restPositions=restPosition,
                        color=color,
                    )
                    correspondingBodyNodes.append(self.skel.getNumBodyNodes())
                    i += 1

                # add information of the corresponding bodyNodes to the branch
                branchInfo["bodyNodeIndices"] = correspondingBodyNodes
                branch.setBranchInfo(branchInfo)

                # add information of the corresponding bodyNode to the endNode
                endNode = branch.getEndNode()
                endNode.setNodeInfo({"bodyNodeIndex": correspondingBodyNodes[-1]})
                if self.topology.isLeafNode(endNode) == True:
                    leafNode = self.topology.getLeafNodeFromNode(endNode)
                    leafNode.setLeafNodeInfo(
                        {"bodyNodeIndex": correspondingBodyNodes[-1]}
                    )
                elif self.topology.isBranchNode(endNode) == True:
                    branchNode = self.topology.getBranchNodeFromNode(endNode)
                    branchNode.setBranchNodeInfo(
                        {"bodyNodeIndex": correspondingBodyNodes[-1]}
                    )

                childBranches = self.topology.getChildBranches(branch)
                if len(childBranches) > 0:
                    for childBranch in self.topology.getChildBranches(branch):
                        nextBranchCandidates.append(childBranch)
                unvisitedBranches.remove(branch)
                blacklist.append(branch)

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
