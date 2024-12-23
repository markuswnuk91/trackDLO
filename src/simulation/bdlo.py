import os, sys
import numpy as np
import math
import dartpy as dart
from warnings import warn
from scipy.spatial.transform import Rotation as R

try:
    sys.path.append(os.getcwd().replace("/src/simulation", ""))
    from src.simulation.dlo import DeformableLinearObject
    from src.modelling.topologyModel import topologyModel
    from src.utils.utils import dampedPseudoInverse
    from src.visualization.colors import thesisColorPalettes
except:
    print("Imports for DLO failed.")


class BDLOTopology(topologyModel):
    """
    A bdloSpecification is topological description of a BDLO which contains information about the branches of the BDLO which are requried to build a kinematic model.
    The class is drived from a topologicalTree but requires additional information to be stored for each branch.
    The additional information is stored as dict in the branchInfo of each branch.

    Attributes:
    branchSpecs (list of dict): branch specifications for each branch of the BDLO as a dict of parameters. For the required parameters see: branchSpec
    branchSpec (dict) with parameters as keywords:
        - radius (float): radius of the branch [m]
        - numSegments (int): desired number of segments the branch should be discretized into
        - density (float): density of the DLO material [kg/m^3]
        - color (np.array): color of the branch [RGB Values]
        - bendingStiffness (float): bending stiffness of the branch [N/rad]
        - torsionalStiffness (float): torsional stiffness of the branch [N/rad]
        - bendingDampingCoeffs (float): bending damping coefficients of the branch [Ns/rad]
        - torsionalDampingCoeffs (float): torsional damping coefficients of the branch[Ns/rad]
        - rootJointRestPositions (np.array): rest position of the branch in [Rx, Ry, Rz] as angles of the ballJoint in bodyNode coodinates at the branch point [rad]
    """

    def __init__(
        self,
        branchSpecs: list = None,
        defaultNumBodyNodes=30,
        defaultRadius=0.01,
        defaultDensity=1000,
        defaultColor=[0, 0, 1],
        defaultBendingStiffness=1,
        defaultTorsionalStiffness=1,
        defaultBendingDampingCoeff=0.1,
        defaultTorsionalDampingCoeff=0.1,
        verbose=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.verbose = False if verbose is None else verbose

        if branchSpecs is None:
            if self.verbose:
                warn("No branch specifications provided. Using default values.")
            self.branchSpecs = [{}] * len(self.branches)
        else:
            self.branchSpecs = branchSpecs

        # make sure specification contains all necessary information
        for i, branchSpec in enumerate(self.branchSpecs):
            if "radius" not in branchSpec:
                if self.verbose:
                    warn(
                        "Expected the branch radius to be specified in the branch specification, but specification has no parameter radius for branch {}. Assuming default value for radius.".format(
                            i
                        )
                    )
                newSpec = self.branchSpecs[i].copy()
                newSpec["radius"] = defaultRadius
                self.branchSpecs[i] = newSpec

            if "density" not in branchSpec:
                if self.verbose:
                    warn(
                        "Expected the branch denisty to be specified in the branch specification, but specification has no parameter density for branch {}. Assuming default value of for density ({} kg/m^3).".format(
                            i, defaultDensity
                        )
                    )
                newSpec = self.branchSpecs[i].copy()
                newSpec["density"] = defaultDensity
                self.branchSpecs[i] = newSpec

            if "numSegments" not in branchSpec:
                if self.verbose:
                    warn(
                        "Expected the desired number of segments to be specified in the branch specification, but specification has no parameter numSegments for branch {}.".format(
                            i
                        )
                    )
                newSpec = self.branchSpecs[i].copy()
                newSpec["numSegments"] = int(
                    math.ceil(
                        self.branches[i].getBranchInfo()["length"]
                        / self.getSummedLength()
                        * defaultNumBodyNodes
                    )
                )
                self.branchSpecs[i] = newSpec

            if "color" not in branchSpec:
                if self.verbose:
                    warn(
                        "No color information given for branch {} using default color blue ([0,0,1]).".format(
                            i
                        )
                    )
                newSpec = self.branchSpecs[i].copy()
                newSpec["color"] = defaultColor
                self.branchSpecs[i] = newSpec

            if "bendingStiffness" not in branchSpec:
                if self.verbose:
                    warn(
                        "No bending stiffness information given for branch {} using default stiffness (1 N/rad).".format(
                            i
                        )
                    )
                newSpec = self.branchSpecs[i].copy()
                newSpec["bendingStiffness"] = defaultBendingStiffness
                self.branchSpecs[i] = newSpec

            if "torsionalStiffness" not in branchSpec:
                if self.verbose:
                    warn(
                        "No torsional stiffness information given for branch {} using default stiffness (1 N/rad).".format(
                            i
                        )
                    )
                newSpec = self.branchSpecs[i].copy()
                newSpec["torsionalStiffness"] = defaultTorsionalStiffness
                self.branchSpecs[i] = newSpec

            if "bendingDampingCoeffs" not in branchSpec:
                if self.verbose:
                    warn(
                        "No bending damping coefficient information given for branch {} using default stiffness (0.1 N/rad).".format(
                            i
                        )
                    )
                newSpec = self.branchSpecs[i].copy()
                newSpec["bendingDampingCoeffs"] = defaultBendingDampingCoeff
                self.branchSpecs[i] = newSpec

            if "torsionalDampingCoeffs" not in branchSpec:
                if self.verbose:
                    warn(
                        "No torsional damping coefficient information given for branch {} using default stiffness (0.1 N/rad).".format(
                            i
                        )
                    )
                newSpec = self.branchSpecs[i].copy()
                newSpec["torsionalDampingCoeffs"] = defaultTorsionalDampingCoeff
                self.branchSpecs[i] = newSpec

            if "rootJointRestPositions" not in branchSpec:
                if self.verbose:
                    warn(
                        "No rootJointRestPositions information given for branch {} using default rest position.".format(
                            i
                        )
                    )
                newSpec = self.branchSpecs[i].copy()

                if self.branches[i] == self.rootBranch:
                    # leaf the root at its original position
                    newSpec["rootJointRestPositions"] = np.array([0, 0, 0, 0, 0, 0])
                elif (
                    self.getBranchNodeFromNode(
                        self.branches[i].getStartNode()
                    ).getNumBranches()
                    == 1
                ):
                    newSpec["rootJointRestPositions"] = np.array([0, 0, 0])
                else:
                    thisbranch = self.branches[i]
                    siblingBranches = self.getChildBranches(
                        self.getParentBranch(thisbranch)
                    )
                    numBranches = len(siblingBranches)
                    k = siblingBranches.index(thisbranch)
                    restAngle = (-1) ** k * k * 120 / 180 * math.pi / numBranches
                    # initRestAngle = -60 / 180 * math.pi
                    # deltaAngle = 120 / 180 * math.pi / (numBranches - 1)
                    # restAngle = initRestAngle + k * deltaAngle

                    if i == 0 and self.getNumBranches() > 1:
                        newSpec["rootJointRestPositions"] = np.array(
                            [restAngle, 0, 0, 0, 0, 0]
                        )
                    elif self.getNumBranches() < 1:
                        raise ValueError("Given Topology has no branches.")
                    else:
                        newSpec["rootJointRestPositions"] = np.array([restAngle, 0, 0])

                self.branchSpecs[i] = newSpec

            # acquire length information from topology model
            newSpec = self.branchSpecs[i].copy()
            newSpec["length"] = self.branches[i].getBranchInfo()["length"]
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


class BranchedDeformableLinearObject(BDLOTopology):
    """
    Class implementing a interface for handling Branched Defromable Linear Objects (BDLO) with dart's skeleton class.
    The class consists of a topologyModel descring its topology and a dartSkeleton which can be used for simulation.
    """

    ID = 0

    def __init__(
        self,
        name=None,
        gravity: bool = None,
        collidable: bool = None,
        adjacentBodyCheck: bool = False,
        enableSelfCollisionCheck: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.ID = BranchedDeformableLinearObject.ID
        BranchedDeformableLinearObject.ID += 1
        if name is None:
            self.name = "BDLO_" + str(self.ID)
        else:
            self.name = name

        self.gravity = True if gravity is None else gravity
        self.collidable = True if collidable is None else collidable
        self.adjacentBodyCheck = (
            False if adjacentBodyCheck is None else adjacentBodyCheck
        )
        self.enableSelfCollisionCheck = (
            True if enableSelfCollisionCheck else enableSelfCollisionCheck
        )

        self.frames = {}
        self.segmentLengths = []

        # create dartSkeleton and add branches
        visitedBranches = set()
        queuedBranches = []
        queuedBranches.append(self.rootBranch)

        while len(queuedBranches) > 0:
            branch = queuedBranches.pop(0)
            if branch not in visitedBranches:
                # load branch spec
                branchSpec = self.getBranchSpec(branch)

                # generate model
                if branch == self.rootBranch:
                    branchDLO = DeformableLinearObject(**branchSpec)
                    self.skel = branchDLO.skel.clone()
                    correspondingBodyNodeIndices = list(
                        range(0, self.skel.getNumBodyNodes())
                    )
                    segmentLengths = branchDLO.segmentLengths
                else:
                    newbranchDLO = DeformableLinearObject(**branchSpec)
                    newBranchSkel = newbranchDLO.skel.clone()
                    newBranchSkel.setPositions(
                        [0, 1, 2], branchSpec["rootJointRestPositions"]
                    )
                    currentNumBodyNodes = self.skel.getNumBodyNodes()
                    newNumBodyNodes = newBranchSkel.getNumBodyNodes()
                    correspondingBodyNodeIndices = list(
                        range(
                            currentNumBodyNodes, currentNumBodyNodes + newNumBodyNodes
                        )
                    )
                    parentBranch = self.getParentBranch(branch)
                    parentBodyNodeIdx = parentBranch.getBranchInfo()[
                        "correspondingBodyNodeIndices"
                    ][-1]
                    newBranchSkel.getBodyNode(0).moveTo(
                        self.skel.getBodyNode(parentBodyNodeIdx)
                    )
                    segmentLengths = newbranchDLO.segmentLengths
                    # set transform such that new branch is appended at the end of the parentBody:
                    bodyNodeIndexInNewSkel = currentNumBodyNodes
                    branchRootJointInNewSkel = self.skel.getBodyNode(
                        bodyNodeIndexInNewSkel
                    ).getParentJoint()
                    parentBodyNode = (
                        self.skel.getBodyNode(bodyNodeIndexInNewSkel)
                        .getParentJoint()
                        .getParentBodyNode()
                    )
                    parentBodyNodeOffset = (
                        branchRootJointInNewSkel.getParentBodyNode()
                        .getParentJoint()
                        .getTransformFromChildBodyNode()
                        .translation()
                    )
                    transform = (
                        branchRootJointInNewSkel.getTransformFromParentBodyNode()
                    )
                    transform.set_translation(-parentBodyNodeOffset)
                    branchRootJointInNewSkel.setTransformFromParentBodyNode(transform)

                branch.addBranchInfo(
                    "correspondingBodyNodeIndices", correspondingBodyNodeIndices
                )
                branch.addBranchInfo("segmentLengths", segmentLengths)
                for adjacentBranch in self.getChildBranches(branch):
                    queuedBranches.append(adjacentBranch)
            visitedBranches.add(branch)

    def getBranchSpec(self, branch):
        branchInfo = branch.getBranchInfo()
        branchSpec = {
            "numSegments": branchInfo["numSegments"],
            "length": branchInfo["length"],
            "radius": branchInfo["radius"],
            "density": branchInfo["density"],
            "color": branchInfo["color"],
            "bendingStiffness": branchInfo["bendingStiffness"],
            "torsionalStiffness": branchInfo["torsionalStiffness"],
            "bendingDampingCoeffs": branchInfo["bendingDampingCoeffs"],
            "torsionalDampingCoeffs": branchInfo["bendingDampingCoeffs"],
            "rootJointRestPositions": branchInfo["rootJointRestPositions"],
        }
        return branchSpec

    # general functions
    def getNumBodyNodes(self):
        return self.skel.getNumBodyNodes()

    # branch Functions

    def getBranchBodyNodes(self, branchIndex):
        """
        Returns the dart bodyNodes corresponding to a branch
        """
        bodyNodeList = []
        for bodyNodeIndex in self.getBranch(branchIndex).getBranchInfo()[
            "correspondingBodyNodeIndices"
        ]:
            bodyNodeList.append(self.skel.getBodyNode(bodyNodeIndex))
        return bodyNodeList

    def getBranchBodyNodeIndices(self, branchIndex):
        """
        Returns the dart bodyNodeIndices corresponding to a branch
        """
        bodyNodeIndices = self.getBranch(branchIndex).getBranchInfo()[
            "correspondingBodyNodeIndices"
        ]
        return bodyNodeIndices

    def getBranchRootBodyNodeIndex(self, branchIndex):
        """returns the bodyNode index of the first bodyNode of a branch with the given index

        Args:
            branchIndex (int): index of the branch for which the first bodyNode should be determined

        Returns:
            int: index of the bodyNode in the dart skeleton
        """
        rootBodyNodeIndex = self.getBranch(branchIndex).getBranchInfo()[
            "correspondingBodyNodeIndices"
        ][0]
        return rootBodyNodeIndex

    def getBranchLastBodyNodeIndex(self, branchIndex):
        """returns the bodyNode index of the last bodyNode of a branch with the given index

        Args:
            branchIndex (int): index of the branch for which the first bodyNode should be determined

        Returns:
            int: index of the bodyNode in the dart skeleton
        """
        lastBodyNodeIndex = self.getBranch(branchIndex).getBranchInfo()[
            "correspondingBodyNodeIndices"
        ][-1]
        return lastBodyNodeIndex

    def getBranchRootDofIndices(self, branchIndex):
        branchRootDofIndices = []
        branchRootBodyNodeIndex = self.getBranch(branchIndex).getBranchInfo()[
            "correspondingBodyNodeIndices"
        ][0]
        branchRootBodyNode = self.skel.getBodyNode(branchRootBodyNodeIndex)
        brachRootJoint = branchRootBodyNode.getParentJoint()
        numRootJointDofs = brachRootJoint.getNumDofs()
        for i in range(0, numRootJointDofs):
            branchRootDofIndices.append(brachRootJoint.getIndexInSkeleton(i))
        return np.array(branchRootDofIndices)

    def setGeneralizedCoordinates(self, q):
        self.skel.setPositions(q)

    def setBranchRootDof(self, branchIndex, dofIndex, dofValue):
        """sets a root Dof of a branch to the specified value

        Args:
            branchIndex (int): index of the branch
            dofIndex (int): index of the dof (for the joint dofs, e.g. 0 for x, 1 for y, 2 for z)
            dofValue (float): value of the dof
        """
        dofIndexInSkeleton = self.getBranchRootDofIndices(branchIndex)[dofIndex]
        self.skel.setPosition(dofIndexInSkeleton, dofValue)
        return

    def setBranchRootDofs(self, branchIndex, dofValues):
        dofIndicesInSkeleton = self.getBranchRootDofIndices(branchIndex)
        for i, value in enumerate(dofValues):
            self.skel.setPosition(dofIndicesInSkeleton[i], value)

    def _getBodyNodeIndicesFromBranch(self, branch):
        return branch.getBranchInfo()["correspondingBodyNodeIndices"]

    def getJointLocalCoordinatesFromBranch(self, branchIndex):
        branchLength = self.branches[branchIndex].getBranchInfo()["length"]
        segmentLengths = self.branches[branchIndex].getBranchInfo()["segmentLengths"]
        if self.getBranch(branchIndex) == self.rootBranch:
            segmentLengths = segmentLengths[::-1]
        localCoordinates = np.insert(
            np.cumsum(np.array(segmentLengths)) / branchLength, 0, 0
        )
        return localCoordinates

    # cartesian to local space conversion functions
    def getBodyNodeCenterLocalCoordinates(self, bodyNodeIndex):
        correspondingBranchIndex = self.getBranchCorrespondanceForBodyNode(
            bodyNodeIndex
        )
        SJoint = self.getJointLocalCoordinatesFromBranch(correspondingBranchIndex)
        bodyNodeIndicesInBranch = self.getBranch(
            correspondingBranchIndex
        ).getBranchInfo()["correspondingBodyNodeIndices"]
        indexInBranch = np.where(np.array(bodyNodeIndicesInBranch) == bodyNodeIndex)[0][
            0
        ]
        SCenter = SJoint[:-1] + np.diff(SJoint) / 2
        if self.getBranch(correspondingBranchIndex) == self.rootBranch:
            SCenter = SCenter[::-1]
        return SCenter[indexInBranch]

    def getCartesianJointPositions(self):
        """returns the cartesian positions of all joints (including start and end joint)"""
        cartesianJointPositions = []
        for bodyNodeIndex in range(0, self.skel.getNumBodyNodes()):
            bodyNode = self.skel.getBodyNode(bodyNodeIndex)
            if bodyNodeIndex == 0:
                parentJointOffset = (
                    bodyNode.getParentJoint()
                    .getTransformFromChildBodyNode()
                    .translation()
                )
                parentJointPosition = bodyNode.getWorldTransform().multiply(
                    parentJointOffset
                )
                cartesianJointPositions.append(parentJointPosition)

            if bodyNode.getNumChildJoints() == 0:
                childJointOffset = (
                    -1
                ) * bodyNode.getParentJoint().getTransformFromChildBodyNode().translation()
                childJointPosition = bodyNode.getWorldTransform().multiply(
                    childJointOffset
                )
            else:
                childJointOffset = (
                    bodyNode.getChildJoint(0)
                    .getTransformFromParentBodyNode()
                    .translation()
                )
                childJointPosition = bodyNode.getWorldTransform().multiply(
                    childJointOffset
                )
            cartesianJointPositions.append(childJointPosition)
        return np.array(cartesianJointPositions)

    def getCartesianBodyCenterPositions(self, returnBranchLocalCoordinates=False):
        cartesianBodyCenterPositions = []
        branchCorrespondance = []
        S = []
        for bodyNodeIndex in range(0, self.skel.getNumBodyNodes()):
            cartesianBodyCenterPositions.append(
                self.skel.getBodyNode(bodyNodeIndex).getTransform().translation()
            )
            branchCorrespondance.append(
                self.getBranchCorrespondanceForBodyNode(bodyNodeIndex)
            )
            S.append(self.getBodyNodeCenterLocalCoordinates(bodyNodeIndex))
        if returnBranchLocalCoordinates:
            return np.array(cartesianBodyCenterPositions), branchCorrespondance, S
        else:
            return np.array(cartesianBodyCenterPositions)

    # correspondance functions
    def getBranchCorrespondanceForBodyNode(self, bodyNodeIndex):
        correspondingBranches = []
        for branchIndex, branch in enumerate(self.branches):
            if bodyNodeIndex in branch.getBranchInfo()["correspondingBodyNodeIndices"]:
                correspondingBranches.append(branchIndex)
        if len(correspondingBranches) > 1:
            raise ValueError(
                "BodyNode corresponds to more than one branch. Something went wrong in the model generation."
            )
        return correspondingBranches[0]

    def getBranchCorrespondancesForBodyNodes(self):
        correspondingBranchIndices = []
        bodyNodeIndices = []
        for bodyNodeIndex in range(self.skel.getNumBodyNodes()):
            for branchIndex, branch in enumerate(self.branches):
                if (
                    bodyNodeIndex
                    in branch.getBranchInfo()["correspondingBodyNodeIndices"]
                ):
                    bodyNodeIndices.append(bodyNodeIndex)
                    correspondingBranchIndices.append(branchIndex)
        return (bodyNodeIndices, correspondingBranchIndices)

    def getBranchCorrespondancesForJoints(self):
        correspondingBranchIndices = []
        jointIndices = list(range(0, self.skel.getNumBodyNodes() + 1))
        numBranches = self.getNumBranches()
        correspondanceMatrix = np.zeros((len(jointIndices), numBranches))
        for i, bodyNodeIndex in enumerate(range(self.skel.getNumBodyNodes())):
            parentBodyNode = self.skel.getBodyNode(bodyNodeIndex).getParentBodyNode()
            for branchIndex, branch in enumerate(self.branches):
                if (
                    bodyNodeIndex
                    in branch.getBranchInfo()["correspondingBodyNodeIndices"]
                ):
                    if parentBodyNode is not None:
                        parentBodyNodeIndex = parentBodyNode.getIndexInSkeleton()
                        correspondanceMatrix[parentBodyNodeIndex + 1, branchIndex] = 1
                    else:
                        correspondanceMatrix[i, branchIndex] = 1
                    correspondanceMatrix[i + 1, branchIndex] = 1
        for i in range(0, len(correspondanceMatrix)):
            correspondingBranchIndices.append(
                np.where(correspondanceMatrix[i, :] == 1)[0].tolist()[0]
            )
        return (jointIndices, correspondingBranchIndices, correspondanceMatrix)

    def getBranchCorrespondancesForSegmentCenters(self):
        segmentIndices, correspondingBranchIndices, correspondanceMatrix = (
            self.getBranchCorrespondancesForJoints()
        )
        segmentIndices = np.delete(segmentIndices, (0), axis=0)
        correspondingBranchIndices = np.delete(correspondingBranchIndices, (0), axis=0)
        correspondanceMatrix = np.delete(correspondanceMatrix, (0), axis=0)
        return (segmentIndices, correspondingBranchIndices, correspondanceMatrix)

    # custom functions for plotting
    def getAdjacentPointPairs(self, q=None):
        if q is not None:
            self.setGeneralizedCoordinates(q)
        pointPairs = []
        for bodyNodeIndex in range(0, self.skel.getNumBodyNodes()):
            jointPositions = self.getCartesianJointPositionsForBodyNode(bodyNodeIndex)
            pointPair = (jointPositions[0], jointPositions[1])
            pointPairs.append(pointPair)
        return pointPairs

    # custom functions for plotting
    def getAdjacencyMatrix(self):
        adjacencyMatrix = np.zeros(
            (self.skel.getNumBodyNodes() + 1, self.skel.getNumBodyNodes() + 1)
        )
        for i in range(self.skel.getNumBodyNodes()):
            bodyNode = self.skel.getBodyNode(i)
            parentBodyNode = bodyNode.getParentBodyNode()
            if parentBodyNode is None:
                adjacencyMatrix[i, i + 1] = 1
                adjacencyMatrix[i + 1, i] = 1
            else:
                j = parentBodyNode.getIndexInSkeleton()
                adjacencyMatrix[i + 1, j + 1] = 1
        return adjacencyMatrix

    def getJointPositionsAndAdjacencyMatrix(self, q=None):
        if q is not None:
            self.setGeneralizedCoordinates(q)
        pointPairs = []
        jointIdx = 0
        points = []
        for bodyNodeIndex in range(0, self.skel.getNumBodyNodes()):
            jointPositions = self.getCartesianJointPositionsForBodyNode(bodyNodeIndex)
            pointPair = (jointPositions[0], jointPositions[1])
            pointPairs.append(pointPair)
        # pointPairs = []
        # for bodyNodeIndex in range(0, self.skel.getNumBodyNodes()):
        #     jointPositions = self.getCartesianJointPositionsForBodyNode(bodyNodeIndex)
        #     pointPair = (jointPositions[0], jointPositions[1])
        #     pointPairs.append(pointPair)
        # points = np.unique(
        #     np.array([point for pointPair in pointPairs for point in pointPair]), axis=0
        # )
        # # adjacencyMatrix = np.zeros((len(points), len(points)))
        # # for pointPair in pointPairs:
        # #     for idx, point in enumerate(points):
        # #         if np.array_equal(pointPair[0], point):
        # #             firstPointIdx = idx
        # #         if np.array_equal(pointPair[1], point):
        # #             secondPointIdx = idx
        # #     adjacencyMatrix[firstPointIdx, secondPointIdx] = 1
        # adjacencyMatrix = np.zeros((len(points), len(points)))
        # for i, point in enumerate(points):
        #     for pointPair in pointPairs:
        #         if np.array_equal(point, pointPair[0]):

        #     for j, otherPoint in enumerate(points):
        #         for pointPair in pointPairs:
        #             firstInTuple = any(
        #                 np.array_equal(point, thisPoint) for point in pointPair
        #             )
        #             secondInTuple = any(
        #                 np.array_equal(point, otherPoint) for point in pointPair
        #             )
        #             if firstInTuple and secondInTuple:
        #                 adjacencyMatrix[i, j] = 1
        jointPositions = []
        for i in range(self.skel.getNumBodyNodes()):
            bodyNode = self.skel.getBodyNode(i)
            parentBodyNode = bodyNode.getParentBodyNode()
            if parentBodyNode is None:
                jointPositions.append(self.getCartesianPositionSegmentStart(i))
                jointPositions.append(self.getCartesianPositionSegmentEnd(i))
            else:
                jointPositions.append(self.getCartesianPositionSegmentEnd(i))
        adjacencyMatrix = self.getAdjacencyMatrix()

        # for bodyNodeIndex in range(0, self.skel.getNumBodyNodes()):
        #     startJointPosition = self.getCartesianPositionSegmentStart(bodyNodeIndex)
        #     if
        #     pointPair = (jointPositions[0], jointPositions[1])
        return np.vstack(jointPositions), adjacencyMatrix

    def getAdjacentPointPairsAndBranchCorrespondance(self):
        pointPairs = []
        for bodyNodeIndex in range(0, self.skel.getNumBodyNodes()):
            jointPositions = self.getCartesianJointPositionsForBodyNode(bodyNodeIndex)
            correspondingBranchIndex = self.getBranchCorrespondanceForBodyNode(
                bodyNodeIndex
            )
            pointPair = (jointPositions[0], jointPositions[1], correspondingBranchIndex)
            pointPairs.append(pointPair)
        return pointPairs

    # position functions
    def getCartesianJointPositionsForBodyNode(self, bodyNodeIndex):
        cartesianJointPositions = []
        bodyNode = self.skel.getBodyNode(bodyNodeIndex)
        parentJointOffset = (
            bodyNode.getParentJoint().getTransformFromChildBodyNode().translation()
        )
        parentJointPosition = bodyNode.getWorldTransform().multiply(parentJointOffset)
        cartesianJointPositions.append(parentJointPosition)

        if bodyNode.getNumChildJoints() == 0:
            childJointOffset = (
                -1
            ) * bodyNode.getParentJoint().getTransformFromChildBodyNode().translation()
            childJointPosition = bodyNode.getWorldTransform().multiply(childJointOffset)
        else:
            childJointOffset = (
                bodyNode.getChildJoint(0).getTransformFromParentBodyNode().translation()
            )
            childJointPosition = bodyNode.getWorldTransform().multiply(childJointOffset)
        cartesianJointPositions.append(childJointPosition)
        return np.array(cartesianJointPositions)

    # custom functions for pose optimnization
    def getCartesianPositionSegmentStart(self, bodyNodeIndex: int):
        """returns the cartesian position of the beginning of a segment

        Args:
            bodyNodeIndex (int): the index of the bodyNode

        Returns:
            np.array: cartesian position of the start of the segement
        """
        bodyNodeTransform = (
            self.skel.getBodyNode(bodyNodeIndex).getWorldTransform().matrix()
        )
        relativeTransformToParentJoint = (
            self.skel.getBodyNode(bodyNodeIndex)
            .getParentJoint()
            .getTransformFromChildBodyNode()
            .matrix()
        )
        return (bodyNodeTransform @ relativeTransformToParentJoint)[:3, 3]

    def getCartesianPositionSegmentEnd(self, bodyNodeIndex: int):
        """returns the cartesian position of the end of a segment

        Args:
            bodyNodeIndex (int): the index of the bodyNode

        Returns:
            np.array: cartesian position of the start of the segement
        """
        if bodyNodeIndex == -1:
            bodyNodeIndex = self.skel.getNumBodyNodes() - 1

        bodyNodeTransform = (
            self.skel.getBodyNode(bodyNodeIndex).getWorldTransform().matrix()
        )
        relativeTransformToParentJoint = (
            self.skel.getBodyNode(bodyNodeIndex)
            .getParentJoint()
            .getTransformFromChildBodyNode()
            .matrix()
        )
        # reverse direction to go to end of segment
        relativeTransformToParentJoint[:3, 3] = -relativeTransformToParentJoint[:3, 3]
        return (bodyNodeTransform @ relativeTransformToParentJoint)[:3, 3]

    def getBodyNodeIndexFromBranchLocalCoodinate(self, branchIndex: int, s: float):
        """returns the bodyNode index corresponding to the local coordinate running along a branch. Local coordinate runs from branch startNode to branch end node, except for rootBranch where it runs from end node to start node, because rootBranch starts with a leafnode.

        Args:
            s (float): local coordinate of the branch in [0,1]

        Returns:
            int: bodyNode index of the body the local coordinate corresponds to.
        """
        if (self.getBranch(branchIndex) == self.rootBranch) and (
            1 - s <= np.finfo(float).eps
        ):
            return self.getBranchRootBodyNodeIndex(branchIndex)
        elif (self.getBranch(branchIndex) == self.rootBranch) and (
            s <= np.finfo(float).eps
        ):
            return self.getBranchLastBodyNodeIndex(branchIndex)

        elif 1 - s <= np.finfo(float).eps:
            return self.getBranchLastBodyNodeIndex(branchIndex)
        elif s <= np.finfo(float).eps:
            return self.getBranchRootBodyNodeIndex(branchIndex)
        else:
            bodyNodeIndicesInBranch = self.branches[branchIndex].getBranchInfo()[
                "correspondingBodyNodeIndices"
            ]
            jointLocalCoordinates = self.getJointLocalCoordinatesFromBranch(branchIndex)
            indexInBranch = (
                next(
                    index[0]
                    for index in enumerate(jointLocalCoordinates)
                    if index[1] > s
                )
                - 1
            )
            if self.getBranch(branchIndex) == self.rootBranch:
                bodyNodeIndex = bodyNodeIndicesInBranch[-(indexInBranch + 1)]
            else:
                bodyNodeIndex = bodyNodeIndicesInBranch[indexInBranch]
            return bodyNodeIndex

    def getOffsetInBodyNodeCoordinatesFromBranchLocalCoordiate(
        self, branchIndex: int, bodyNodeIndex, s: float
    ):
        """returns the offset from a center of a bodyNode to the location corresponding to a local coordinate.
        The offset is expressed in the cooresponding local bodyNode frame

        Args:
            branchIndex: index of the corresponding branch
            bodyNodeIndex: index of the corresponding bodyNode
            s (float): local coordinate in [0,1]

        Returns:
            np.array: 3x1 offset vector fom the body node center to the position corresponding to the given local coordinate.
        """
        bodyNodeIndicesInBranch = self.branches[branchIndex].getBranchInfo()[
            "correspondingBodyNodeIndices"
        ]
        segmentLengths = self.branches[branchIndex].getBranchInfo()["segmentLengths"]
        localCoordsJoints = self.getJointLocalCoordinatesFromBranch(branchIndex)
        indexInBranch = bodyNodeIndicesInBranch.index(bodyNodeIndex)
        if self.getBranch(branchIndex) == self.rootBranch:
            sUpper = localCoordsJoints[-(indexInBranch + 1)]
            sLower = localCoordsJoints[-(indexInBranch + 2)]
            sCenter = sLower + (sUpper - sLower) / 2
            sOffset = (s - sCenter) / (sUpper - sLower)
            offset = np.array([0, 0, -sOffset * segmentLengths[indexInBranch]])
        else:
            sLower = localCoordsJoints[indexInBranch]
            sUpper = localCoordsJoints[indexInBranch + 1]
            sCenter = sLower + (sUpper - sLower) / 2
            sOffset = (s - sCenter) / (sUpper - sLower)
            offset = np.array([0, 0, sOffset * segmentLengths[indexInBranch]])
        return offset

    def getCartesianPositionSegmentWithOffset(
        self, bodyNodeIndex: int, offset: np.array
    ):
        """returns the cartesian position of the center of a segment

        Args:
            bodyNodeIndex (int): the index of the bodyNode

        Returns:
            np.array: cartesian position of the center of the segement
        """
        return (
            self.skel.getBodyNode(bodyNodeIndex).getWorldTransform().matrix()
            @ np.append(offset, 1)
        )[:3]

    def getCartesianPositionFromBranchLocalCoordinate(self, branchIndex: int, s: float):
        """returns the cartesian position of a point along a branch of the bdlo specified by a local coordinate running along the branch from the startNode to the endNode

        Args:
            branchIndex (int): index of the branch
            s (float): local coordinate for the branch in [0,1] where 0 corresponds to the start of the branch and 1 corresponds to the end of the branch

        Returns:
            cartesianPosition: cartesian position of the point corresponding to the local coordinate.
        """
        if self.getBranch(branchIndex) == self.rootBranch and s <= np.finfo(float).eps:
            correspondingBodyNodeIndex = self.getBranchLastBodyNodeIndex(branchIndex)
            return self.getCartesianPositionSegmentEnd(correspondingBodyNodeIndex)
        elif (
            self.getBranch(branchIndex) == self.rootBranch
            and 1 - s <= np.finfo(float).eps
        ):
            correspondingBodyNodeIndex = self.getBranchRootBodyNodeIndex(branchIndex)
            return self.getCartesianPositionSegmentStart(correspondingBodyNodeIndex)
        elif s <= np.finfo(float).eps:
            correspondingBodyNodeIndex = self.getBranchRootBodyNodeIndex(branchIndex)
            return self.getCartesianPositionSegmentStart(correspondingBodyNodeIndex)
        elif 1 - s <= np.finfo(float).eps:
            correspondingBodyNodeIndex = self.getBranchLastBodyNodeIndex(branchIndex)
            return self.getCartesianPositionSegmentEnd(correspondingBodyNodeIndex)
        else:
            correspondBodyNodeIdx = self.getBodyNodeIndexFromBranchLocalCoodinate(
                branchIndex, s
            )
            offset = self.getOffsetInBodyNodeCoordinatesFromBranchLocalCoordiate(
                branchIndex, correspondBodyNodeIdx, s
            )
            return self.getCartesianPositionSegmentWithOffset(
                correspondBodyNodeIdx, offset
            )

    def getCartesianPositionsFromBranchLocalCoordinates(self, branchIndex: int, S):
        """Returns the cartesian positions for the given  local coordinates in a branch

        Args:
            branchIndex (int): index of the  branch for which the cartesian positions should be retrived
            S (np.array): local coordinates corresponding to the cartesian positions, measured from startNode of the branch to the end node of the branch
        """
        X = np.zeros((S.size, 3))
        for i, s in enumerate(S):
            X[i, :] = self.getCartesianPositionFromBranchLocalCoordinate(branchIndex, s)
        return X

    def getJacobianFromBranchLocalCoordinate(self, branchIndex: int, s: float):
        correspondingBodyNode = self.getBodyNodeIndexFromBranchLocalCoodinate(
            branchIndex, s
        )
        offset = self.getOffsetInBodyNodeCoordinatesFromBranchLocalCoordiate(
            branchIndex, correspondingBodyNode, s
        )
        # bodyNode = self.skel.getBodyNode(correspondingBodyNode)
        # jacobian = bodyNode.getWorldJacobian(offset)
        # jacobianTrans = jacobian[3:6, :]
        # jacobianRot = jacobian[:3, :]
        # # transformToWorld = np.linalg.inv(
        # #     self.skel.getBodyNode(branchIndex).getWorldTransform().rotation()
        # # )
        # # jacobian = np.vstack(
        # #     (transformToWorld @ jacobianRot, transformToWorld @ jacobianTrans)
        # # )

        # indexPointer = 0
        # paddedJacobian = np.zeros((6, self.skel.getNumDofs()))
        # for i in range(0, self.skel.getNumDofs()):
        #     if bodyNode.dependsOn(i):
        #         paddedJacobian[:, i] = jacobian[:, indexPointer]
        #         indexPointer += 1
        return self.skel.getWorldJacobian(
            self.skel.getBodyNode(correspondingBodyNode), offset
        )

    def getLeafNodeCartesianPositions(self):
        """returns the cartesian positions for all leafnodes"""
        leafNodeCartesiantPositions = []
        for i, branch in enumerate(self.branches):
            if self.getNumLeafNodesFromBranch(branch) == 1:
                cartesianPosition = self.getCartesianPositionFromBranchLocalCoordinate(
                    i, 1
                )
                leafNodeCartesiantPositions.append(cartesianPosition)
        return np.vstack(leafNodeCartesiantPositions)

    def getBranchNodeCartesianPositions(self):
        """returns the cartesian positions for all branchnodes"""
        branchNodeCartesiantPositions = []
        for i, branch in enumerate(self.branches):
            cartesianPosition = self.getCartesianPositionFromBranchLocalCoordinate(i, 0)
            branchNodeCartesiantPositions.append(cartesianPosition)
        return np.vstack(branchNodeCartesiantPositions)

    def getGeneralizedCoordinates(self):
        return self.skel.getPositions()

    def computeForwardKinematics(
        self, q, locations="center", returnBranchLocalCoordinates=True
    ):
        self.skel.setPositions(q)

        if locations == "center":
            return self.getCartesianBodyCenterPositions(returnBranchLocalCoordinates)
        elif locations == "joint":
            return self.getCartesianJointPositions()
        else:
            raise NotImplementedError

    def computeForwardKinematicsFromBranchLocalCoordinates(
        self, q, branchLocalCoordinates
    ):
        """computes the forward kinematics for given joint angel vector q at locations specified by a list of branch local coordinates

        Args:
            q (Nx1 np.array): generalized coordinates of the skeleton
            branchLocalCoordinates (list of tuple): branch local coordinates as tuples of (branchIndex, s)
        """
        self.setGeneralizedCoordinates(q)
        X = []
        for branchLocalCoordinate in branchLocalCoordinates:
            x = self.getCartesianPositionFromBranchLocalCoordinate(
                branchLocalCoordinate[0], branchLocalCoordinate[1]
            )
            X.append(x)
        return np.array(X)

    def getBodyNodeNodeAdjacencyMatrix(self):
        bodyNodeAdjacencyMatrix = np.zeros(
            (self.skel.getNumBodyNodes(), self.skel.getNumBodyNodes())
        )
        for i, bodyNode in enumerate(self.skel.getBodyNodes()):
            parentBodyNode = bodyNode.getParentBodyNode()
            numChildBodyNodes = bodyNode.getNumChildBodyNodes()
            if parentBodyNode is not None:
                j = parentBodyNode.getIndexInSkeleton()
                bodyNodeAdjacencyMatrix[i, j] = 1
            if numChildBodyNodes > 0:
                for childBodyNodeIndex in range(0, numChildBodyNodes):
                    childBodyNode = bodyNode.getChildBodyNode(childBodyNodeIndex)
                    j = childBodyNode.getIndexInSkeleton()
                    bodyNodeAdjacencyMatrix[i, j] = 1
        return bodyNodeAdjacencyMatrix

    def computeInverseKinematics(
        self,
        targets,
        numIterations=100,
        qInit=None,
        method="damped",
        damping=None,
        targetPositions="centers",
        verbose=False,
    ):
        qInit = self.skel.getPositions() if qInit is None else qInit
        damping = 1 if damping is None else damping
        q = qInit.copy()[:, None]
        dq = np.zeros((self.skel.getNumDofs(), 1))
        self.skel.setPositions(qInit)
        if method == "damped":
            iteration = 0
            N = self.skel.getNumBodyNodes()
            Dof = self.skel.getNumDofs()
            jacobians = np.zeros((3 * N, Dof))
            errors = np.zeros((3 * N, 1))
            currentPositions = np.zeros((3 * N, 1))
            targetPositions = targets.flatten()[:, None]
            while iteration < numIterations:
                self.skel.setPositions(q)
                for i, bodyNode in enumerate(self.skel.getBodyNodes()):
                    currentPositions[3 * i : 3 * i + 3] = (
                        bodyNode.getTransform().translation()[:, None]
                    )
                    jacobians[3 * i : 3 * i + 3, :] = self.skel.getWorldJacobian(
                        bodyNode, np.array((0, 0, 0))
                    )[3:6, :]
                errors = targetPositions - currentPositions
                dq = dampedPseudoInverse(jacobians, damping) @ (errors)
                q = q + dq
                iteration += 1
                if verbose:
                    print("Iteration: {}/{}".format(iteration, numIterations))
        else:
            raise NotImplementedError
        return q.flatten()

    def samplePointsForCorrespondanceEstimation(self, q, S):
        # sample points from templateTopology
        B = []
        cartesianPositions = []
        for branch in self.getBranches():
            if self.isOuterBranch(branch):
                branchIndex = self.getBranchIndex(branch)
                # get cartesian positions
                for s in S:
                    cartesianPosition = (
                        self.getCartesianPositionFromBranchLocalCoordinate(
                            branchIndex, s
                        )
                    )
                    cartesianPositions.append(cartesianPosition)
                    B.append(branchIndex)
            elif self.isInnerBranch(branch):
                branchIndex = self.getBranchIndex(branch)
                # get cartesian positions
                s = 0.5
                cartesianPosition = self.getCartesianPositionFromBranchLocalCoordinate(
                    branchIndex, s
                )
                cartesianPositions.append(cartesianPosition)
                B.append(branchIndex)
        return np.array(cartesianPositions), B

    def convertRotVecToBallJointPositions(self, angle, axis):
        rotMat = R.from_rotvec(angle * axis).as_matrix()
        return dart.dynamics.BallJoint.convertToPositions(rotMat)

    def convertExtrinsicEulerAnglesToBallJointPositions(
        self, xRotAngle, yRotAngle, zRotAngle, degreesInRad=True
    ):
        if degreesInRad:
            rotMat = R.from_euler("xyz", [xRotAngle, yRotAngle, zRotAngle]).as_matrix()
        else:
            raise NotImplementedError
        return dart.dynamics.BallJoint.convertToPositions(rotMat)

    def convertIntrinsicEulerAnglesToBallJointPositions(
        self, xRotAngle, yRotAngle, zRotAngle, degreesInRad=True
    ):
        if degreesInRad:
            rotMat = R.from_euler("XYZ", [xRotAngle, yRotAngle, zRotAngle]).as_matrix()
        else:
            raise NotImplementedError
        return dart.dynamics.BallJoint.convertToPositions(rotMat)

    def setInitialPose(
        self, initialPosition=np.array([0, 0, 0]), initialRotation=np.array([0, 0, 0])
    ):
        for i, pos in enumerate(initialPosition):
            self.skel.setPosition(i + 3, pos)

        q_rot = self.convertIntrinsicEulerAnglesToBallJointPositions(
            initialRotation[0], initialRotation[1], initialRotation[2]
        )
        for i, rot in enumerate(q_rot):
            q = self.skel.getPosition(i)
            self.skel.setPosition(i, q + rot)
        return

    def getSamplePositionsFromLocalCoordinates(self, S):
        K = self.getNumBranches()
        XSamples = []
        for b in range(0, K):
            if self.isOuterBranch(self.getBranch(b)):
                for s in S:
                    x = self.getCartesianPositionFromBranchLocalCoordinate(b, s)
                    XSamples.append(x)
            elif self.isInnerBranch(self.getBranch(b)):
                s = 0.5
                x = self.getCartesianPositionFromBranchLocalCoordinate(b, s)
                XSamples.append(x)
        return np.array(XSamples)

    def setColor(self, color):
        bodyNodes = self.skel.getBodyNodes()
        for bodyNode in bodyNodes:
            bodyNode.getShapeNode(0).getVisualAspect().setColor(color)
            bodyNode.getShapeNode(1).getVisualAspect().setColor(color)
        return

    def setColorForBranch(self, branchIndex, color):
        branchBodyNodeIndices = self.getBranchBodyNodeIndices(branchIndex)
        for bodyNodeIdx in branchBodyNodeIndices:
            bodyNode = self.skel.getBodyNode(bodyNodeIdx)
            bodyNode.getShapeNode(0).getVisualAspect().setColor(color)
            bodyNode.getShapeNode(1).getVisualAspect().setColor(color)
        return

    def setBranchColorsFromColorPalette(self, colorPalette=None):
        colorPalette = (
            thesisColorPalettes["viridis"] if colorPalette is None else colorPalette
        )
        branchIndices = self.getBranchIndices(self.getBranches())
        numBranches = len(branchIndices)
        colorScaleCoordinates = np.linspace(0, 1, numBranches)
        branchColors = []
        for s in colorScaleCoordinates:
            branchColors.append(colorPalette.to_rgba(s)[:3])
        for branchIndex, branchColors in zip(branchIndices, branchColors):
            self.setColorForBranch(branchIndex, branchColors)
        return

    def setStiffnessForAllDof(self, stiffness):
        for i in range(0, self.skel.getNumJoints()):
            joint = self.skel.getJoint(i)
            for j in range(0, joint.getNumDofs()):
                joint.setSpringStiffness(j, stiffness)
        return

    def setDampingForAllDof(self, damping):
        for i in range(0, self.skel.getNumJoints()):
            joint = self.skel.getJoint(i)
            for j in range(0, joint.getNumDofs()):
                joint.setDampingCoefficient(j, damping)
        return


# class BranchedDeformableLinearObject(DeformableLinearObject):
#     """
#     Class implementing a interface for handling Branched Defromable Linear Objects (BDLO) with dart's skeleton class.
#     The class consists of a topologyModel descring its topology and a dartSkeleton which can be used for simulation.

#     Attributes:
#         name (str): name of the BDLO
#         skel: dart sekelton belongig to the BDLO
#         topology: topologyModel belonging to the BDLO
#     """

#     ID = 0

#     def __init__(
#         self,
#         topology: BDLOTopology,
#         name=None,
#         gravity: bool = True,
#         collidable: bool = True,
#         adjacentBodyCheck: bool = False,
#         enableSelfCollisionCheck: bool = True,
#     ):

#         self.ID = BranchedDeformableLinearObject.ID
#         BranchedDeformableLinearObject.ID += 1

#         if name is None:
#             self.name = "BDLO_" + str(self.ID)
#         else:
#             self.name = name

#         self.topology = topology

#         self.adjacentBodyCheck = adjacentBodyCheck
#         self.enableSelfCollisionCheck = enableSelfCollisionCheck
#         self.frames = {}
#         self.segmentLengths = []

#         if gravity is None:
#             self.gravity = True
#         else:
#             self.gravity = gravity

#         if collidable is None:
#             self.collidable = True
#         else:
#             self.collidable = collidable


#         # create dartSkeleton
#         if self.topology.getNumBranches() == 1:
#             branchInfo = self.topology.getBranch(0).getBranchInfo()
#             length = branchInfo["length"]
#             radius = branchInfo["radius"]
#             density = branchInfo["density"]
#             numSegments = branchInfo["numSegments"]
#             color = branchInfo["color"]
#             stiffness = np.array(
#                 [
#                     branchInfo["bendingStiffness"],
#                     branchInfo["bendingStiffness"],
#                     branchInfo["torsionalStiffness"],
#                 ]
#             )
#             damping = np.array(
#                 [
#                     branchInfo["bendingDampingCoeffs"],
#                     branchInfo["bendingDampingCoeffs"],
#                     branchInfo["torsionalDampingCoeffs"],
#                 ]
#             )
#             restPositions = branchInfo["restPosition"]
#             segmentLength = length / numSegments
#             super().__init__(
#                 numSegments=numSegments,
#                 length=length,
#                 radius=radius,
#                 density=density,
#                 name=name,
#                 stiffness=stiffness,
#                 damping=damping,
#                 color=color,
#                 gravity=self.gravity,
#                 collidable=self.collidable,
#                 adjacentBodyCheck=self.adjacentBodyCheck,
#                 enableSelfCollisionCheck=self.enableSelfCollisionCheck,
#             )

#         else:
#             self.skel = dart.dynamics.Skeleton(name=self.name)
#             unvisitedBranches = self.topology.getBranches().copy()
#             blacklist = (
#                 []
#             )  # branches for witch bodyNodes were already generated are blacklisted
#             nextBranchCandidates = []

#             # make sure we start with the rootBranch
#             nextBranchCandidates.append(self.topology.getRootBranch())
#             if nextBranchCandidates[0].getStartNode().getParent() is not None:
#                 raise ValueError(
#                     "Expected the first branch to contain the RootNode at its end, but got branch with endode that has parent: ".format(
#                         branch.getStartNode().getParentNode()
#                     )
#                 )

#             # loop to generate the branches
#             while len(unvisitedBranches) > 0:
#                 # get necessary information for generating the bodyNodes for the branch
#                 if len(nextBranchCandidates) > 0:
#                     branch = nextBranchCandidates.pop(0)
#                 else:
#                     branch = unvisitedBranches[0]
#                 branchInfo = branch.getBranchInfo()
#                 length = branchInfo["length"]
#                 radius = branchInfo["radius"]
#                 density = branchInfo["density"]
#                 numSegments = branchInfo["numSegments"]
#                 color = branchInfo["color"]
#                 stiffness = np.array(
#                     [
#                         branchInfo["bendingStiffness"],
#                         branchInfo["bendingStiffness"],
#                         branchInfo["torsionalStiffness"],
#                     ]
#                 )
#                 damping = np.array(
#                     [
#                         branchInfo["bendingDampingCoeffs"],
#                         branchInfo["bendingDampingCoeffs"],
#                         branchInfo["torsionalDampingCoeffs"],
#                     ]
#                 )
#                 restPositions = branchInfo["restPosition"]
#                 segmentLength = length / numSegments
#                 correspondingBodyNodes = []

#                 if (
#                     # make sure we start at the rootBranch
#                     branch == topology.rootBranch
#                     and branch.getStartNode().getParent() is None
#                 ):
#                     # generate the rootBranch
#                     self.makeRootBody(
#                         segmentLength=segmentLength,
#                         radius=radius,
#                         density=density,
#                         restPositions=restPositions,
#                         color=color,
#                     )
#                     correspondingBodyNodes.append(self.skel.getNumBodyNodes() - 1)
#                     for i in range(numSegments - 1):
#                         self.addBody(
#                             parentNode=self.skel.getBodyNodes()[-1],
#                             segmentLength=segmentLength,
#                             radius=radius,
#                             density=density,
#                             stiffnesses=np.ones(3) * stiffness,
#                             dampingCoeffs=np.ones(3) * damping,
#                             restPositions=np.zeros(3),
#                             color=color,
#                         )
#                         correspondingBodyNodes.append(self.skel.getNumBodyNodes() - 1)
#                         i += 1

#                     # add information of the corresponding bodyNodes to the branch
#                     branchInfo["correspondingBodyNodeIndices"] = correspondingBodyNodes
#                     branch.setBranchInfo(branchInfo)

#                     # add information of the corresponding bodyNode to the start node (root)
#                     startNode = branch.getStartNode()
#                     startNode.setNodeInfo({"bodyNodeIndex": correspondingBodyNodes[0]})
#                     if self.topology.isLeafNode(startNode) == True:
#                         leafNode = self.topology.getLeafNodeFromNode(startNode)
#                         leafNode.setLeafNodeInfo(
#                             {"bodyNodeIndex": correspondingBodyNodes[0]}
#                         )
#                     elif self.topology.isBranchNode(startNode) == True:
#                         branchNode = self.topology.getBranchNodeFromNode(startNode)
#                         branchNode.setBranchNodeInfo(
#                             {"bodyNodeIndex": correspondingBodyNodes[0]}
#                         )

#                     # add information of the corresponding bodyNode to the end node (root branch)
#                     endNode = branch.getEndNode()
#                     endNode.setNodeInfo({"bodyNodeIndex": correspondingBodyNodes[-1]})
#                     if self.topology.isLeafNode(endNode) == True:
#                         leafNode = self.topology.getLeafNodeFromNode(endNode)
#                         leafNode.setLeafNodeInfo(
#                             {"bodyNodeIndex": correspondingBodyNodes[-1]}
#                         )
#                     elif self.topology.isBranchNode(endNode) == True:
#                         branchNode = self.topology.getBranchNodeFromNode(endNode)
#                         branchNode.setBranchNodeInfo(
#                             {"bodyNodeIndex": correspondingBodyNodes[-1]}
#                         )

#                     # add information of the corresponding bodyNodes to the memberNodes
#                     for node in branch.getMemberNodes():
#                         localCoord = self.topology.getLocalCoordinateFromBranchNode(
#                             branch, node
#                         )
#                         bodyNodeIndex = self.getBodyNodeFromLocalBranchCoordinate(
#                             branch, localCoord
#                         )
#                         node.setNodeInfo(
#                             {"bodyNodeIndex": correspondingBodyNodes[bodyNodeIndex]}
#                         )

#                 else:
#                     # generate bodyNodes for remaining branches
#                     parentBodyNodeIdx = branch.getStartNode().getNodeInfo()[
#                         "bodyNodeIndex"
#                     ]
#                     parentBodyNode = self.skel.getBodyNode(parentBodyNodeIdx)
#                     for i in range(numSegments):
#                         if i == 0:
#                             self.addBody(
#                                 parentNode=parentBodyNode,
#                                 segmentLength=segmentLength,
#                                 radius=radius,
#                                 density=density,
#                                 stiffnesses=np.ones(3) * stiffness,
#                                 dampingCoeffs=np.ones(3) * damping,
#                                 restPositions=restPositions,
#                                 color=color,
#                                 offset=-segmentLength,
#                             )
#                         else:
#                             self.addBody(
#                                 parentNode=self.skel.getBodyNodes()[-1],
#                                 segmentLength=segmentLength,
#                                 radius=radius,
#                                 density=density,
#                                 stiffnesses=np.ones(3) * stiffness,
#                                 dampingCoeffs=np.ones(3) * damping,
#                                 restPositions=np.zeros(3),
#                                 color=color,
#                             )

#                         correspondingBodyNodes.append(self.skel.getNumBodyNodes() - 1)
#                         i += 1

#                     # add information of the corresponding bodyNodes to the branch
#                     branchInfo["correspondingBodyNodeIndices"] = correspondingBodyNodes
#                     branch.setBranchInfo(branchInfo)

#                     # add information of the corresponding bodyNode to the endNode
#                     endNode = branch.getEndNode()
#                     endNode.setNodeInfo({"bodyNodeIndex": correspondingBodyNodes[-1]})
#                     if self.topology.isLeafNode(endNode) == True:
#                         leafNode = self.topology.getLeafNodeFromNode(endNode)
#                         leafNode.setLeafNodeInfo(
#                             {"bodyNodeIndex": correspondingBodyNodes[-1]}
#                         )
#                     elif self.topology.isBranchNode(endNode) == True:
#                         branchNode = self.topology.getBranchNodeFromNode(endNode)
#                         branchNode.setBranchNodeInfo(
#                             {"bodyNodeIndex": correspondingBodyNodes[-1]}
#                         )

#                     # add information of the corresponding bodyNodes to the memberNodes
#                     for node in branch.getMemberNodes():
#                         localCoord = self.topology.getLocalCoordinateFromBranchNode(
#                             branch, node
#                         )
#                         bodyNodeIndex = self.getBodyNodeFromLocalBranchCoordinate(
#                             branch, localCoord
#                         )
#                         node.setNodeInfo(
#                             {"bodyNodeIndex": correspondingBodyNodes[bodyNodeIndex]}
#                         )

#                 # get next branch candidates for which bodyNodes are generated
#                 siblingBranches = self.topology.getAdjacentBranches(branch)
#                 for siblingBranch in siblingBranches:
#                     if (
#                         siblingBranch not in blacklist
#                         and siblingBranch is not branch
#                         and siblingBranch not in nextBranchCandidates
#                     ):
#                         nextBranchCandidates.append(siblingBranch)
#                 unvisitedBranches.remove(branch)
#                 blacklist.append(branch)

#     def getBranchBodyNodes(self, branchNumber):
#         """
#         Returns the dart bodyNodes corresponding to a branch
#         """
#         bodyNodeList = []
#         for bodyNodeIndex in self.topology.getBranch(branchNumber).getBranchInfo()[
#             "correspondingBodyNodeIndices"
#         ]:
#             bodyNodeList.append(self.skel.getBodyNode(bodyNodeIndex))
#         return bodyNodeList

#     def getBranchBodyNodeIndices(self, branchNumber):
#         """
#         Returns the dart bodyNodes indices corresponding to a branch
#         """
#         correspondingBodyNodeIndices = self.topology.getBranch(branchNumber).getBranchInfo()[
#             "correspondingBodyNodeIndices"
#         ]
#         return correspondingBodyNodeIndices

#     def getLeafBodyNodes(self):
#         """
#         Returns the DART bodyNodes corresponding to the leafNodes in the topology of the BDLO
#         """
#         leafBodyNodes = []
#         for leafNode in self.topology.getLeafNodes():
#             leafBodyNodes.append(
#                 self.skel.getBodyNode(leafNode.getLeafNodeInfo()["bodyNodeIndex"])
#             )
#         return leafBodyNodes

#     def getLeafBodyNodeIndices(self):
#         correspondingBodyNodeIndices = []
#         for leafNode in self.topology.getLeafNodes():
#             correspondingBodyNodeIndices.append(leafNode.getLeafNodeInfo()["bodyNodeIndex"])
#         return correspondingBodyNodeIndices

#     def getBranchPointBodyNodes(self):
#         """
#         Returns the DART bodyNodes corresponding to the branchNodes in the topology of the BDLO
#         """
#         branchBodyNodes = []
#         for branchNode in self.topology.getBranchNodes():
#             branchBodyNodes.append(
#                 self.skel.getBodyNode(branchNode.getBranchNodeInfo()["bodyNodeIndex"])
#             )
#         return branchBodyNodes

#     def getBranchPointBodyNodeIndices(self):
#         branchBodyNodeIndices = []
#         for branchNode in self.topology.getBranchNodes():
#             branchBodyNodeIndices.append(
#                 branchNode.getBranchNodeInfo()["bodyNodeIndex"]
#             )
#         return branchBodyNodeIndices

#     def getBranchIndexFromBodyNodeIndex(self, bodyNodeIndex):
#         """
#         Returns the branch corresponding to a dart bodyNode index.
#         """
#         branchIndex = -1
#         for i, branch in enumerate(self.topology.getBranches()):
#             if bodyNodeIndex in branch.getBranchInfo()["correspondingBodyNodeIndices"]:
#                 return i
#             else:
#                 pass
#         if branchIndex == -1:
#             warn("Given bodyNodeIndex is not in skeleton.")
#             return None

#     def getSegmentLengthFromBranch(self, branchNumber: int):
#         return (
#             self.topology.getBranch(branchNumber).getBranchInfo()["length"]
#             / self.topology.getBranch(branchNumber).getBranchInfo()["numSegments"]
#         )

#     def getBodyNodeFromLocalBranchCoordinate(self, branch, s: float):
#         branchLength = branch.getBranchInfo()["length"]
#         numSegments = branch.getBranchInfo()["numSegments"]
#         segmentLength = branchLength / numSegments
#         if s > 1 or s < 0:
#             raise ValueError(
#                 "Obtained {} as value of the local coordinate. The expected value is from [0, 1]".format(
#                     s
#                 )
#             )
#         else:
#             return int(np.ceil(s * branchLength / segmentLength))

#     def getNumBranches(self):
#         """Returns the number of branches of this BDLO model"""
#         return self.topology.getNumBranches()

#     def getBranchDofIndexInSkel(self, branchPointBodyNodeIndex: int, dof: int):
#         return (
#             self.skel.getBodyNode(branchPointBodyNodeIndex)
#             .getParentJoint()
#             .getIndexInSkeleton(dof)
#         )
# def setBranchDof(self, branchPointIndex: int, dof: int, value: float):
#     branchPointBodyNodeIndex = self.getBranchPointBodyNodeIndices()[
#         branchPointIndex
#     ]
#     dofIdx = self.getBranchDofIndexInSkel(branchPointBodyNodeIndex, dof)
#     self.skel.setPosition(dofIdx, value)
#     return

# def getBranchPointBodyNodeIndices(self):
#     branchBodyNodeIndices = []
#     for branchNode in self.topology.getBranchNodes():
#         branchBodyNodeIndices.append(
#             branchNode.getBranchNodeInfo()["bodyNodeIndex"]
#         )
#     return branchBodyNodeIndices
