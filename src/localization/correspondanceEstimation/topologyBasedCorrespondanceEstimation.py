import sys
import os
import numpy as np
from warnings import warn
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import MinMaxScaler

try:
    sys.path.append(
        os.getcwd().replace("/src/localization/correspondanceEstimation", "")
    )
    from src.localization.topologyExtraction.topologyExtraction import (
        TopologyExtraction,
    )
    from src.simulation.bdlo import BranchedDeformableLinearObject
except:
    print("Imports for Topology-based Correspondance Estimation failed.")
    raise


class TopologyBasedCorrespondanceEstimation(object):
    """Class to perform a topology based correspondance estimation between a given Template Topology of a BDLO and a topology extracted from a given point set

    Attributes:
    Y: numpy.array
        MxD array point cloud representation of the BDLO

    templateTopology: topologyModel
        topology model representing the topology of the BDLO
    """

    def __init__(
        self,
        templateTopology: BranchedDeformableLinearObject,
        Y=None,
        numSeedPoints=None,
        extractedTopology=None,
        *args,
        **kwargs
    ):
        if extractedTopology is None and Y is None:
            raise ValueError(
                "Obtained no extracted topology, nor point set to extract topology from. Please privede a topology or point set."
            )
        elif extractedTopology is None and Y is not None:
            if numSeedPoints is None:
                raise warn(
                    "No number of seedpoints provided for topology extraction, choosing 1/3 of the dataset as default."
                )
                numSeedPoints = int(Y.shape[0] / 3)
            self.topologyExtractor = TopologyExtraction(
                Y=Y, numSeedPoints=numSeedPoints, *args, **kwargs
            )
            X = None

        else:
            numSeedPoints = extractedTopology.X.shape[0]
            X = extractedTopology.X

        self.Y = Y
        self.numSeedPoints = numSeedPoints
        self.templateTopology = templateTopology
        self.X = X
        self.extractedTopology = extractedTopology

    def extractTopology(self):
        if self.extractedTopology is None:
            self.extractedTopology = self.topologyExtractor.extractTopology()
            self.X = self.extractedTopology.X
            if (
                self.extractedTopology.getNumBranches()
                > self.templateTopology.getNumBranches()
            ):
                raise ValueError(
                    "Found more branches than the number of branches in the template topology. Number of extracted branches is: {}, expected number of branches is: {}".format(
                        self.extractedTopology.getNumBranches(),
                        self.templateTopology.getNumBranches(),
                    )
                )
            elif (
                self.extractedTopology.getNumBranches()
                < self.templateTopology.getNumBranches()
            ):
                raise ValueError(
                    "Found less branches than the number of branches in the template topology. Number of extracted branches is: {}, expected number of branches is: {}".format(
                        self.extractedTopology.getNumBranches(),
                        self.templateTopology.getNumBranches(),
                    )
                )
        else:
            warn("Topology is already extracted.")
        return self.extractedTopology

    def getBranchFeatures(self, topology, branch):
        """Determines a feature vector used for correspondance estimation between individual branches

        Args:
            branch (branch): branch for which the featues should be extracted

        Raises:
            NotImplementedError: _description_
        """
        featureList = []

        # branch length
        branchLength = branch.getBranchInfo()["length"]
        featureList.append(branchLength)

        # number of branch poitns
        numBranchPoints = topology.getNumBranchNodesFromBranch(branch)
        featureList.append(numBranchPoints)

        # number of leaf points
        numLeafPoints = topology.getNumLeafNodesFromBranch(branch)
        featureList.append(numLeafPoints)

        # summed length of adjacent branches
        sumAdjacentBranchLength = 0
        adjacentBranches = topology.getAdjacentBranches(branch)
        for adjacentBranch in adjacentBranches:
            sumAdjacentBranchLength += adjacentBranch.getBranchInfo()["length"]
        featureList.append(sumAdjacentBranchLength)

        # length to next branch point
        lengthToNextBranchPoint = 0
        adjacentBranches = topology.getAdjacentBranches(branch)
        for adjacentBranch in adjacentBranches:
            if topology.isBranchNode(
                adjacentBranch.getStartNode()
            ) and topology.isBranchNode(adjacentBranch.getEndNode()):
                lengthToNextBranchPoint += adjacentBranch.getBranchInfo()["length"]
        featureList.append(lengthToNextBranchPoint)

        branchFeatures = np.array(featureList)
        return branchFeatures

    def getCorrespondingBranches(self, templateTopology=None, otherTopology=None):
        """Determines the corresponding branches between the template topology and an other topology

        Returns:
        branchCorrespondence (np.array):
            array of correspondances such that branchCorrespondence[i] is the branch index of the otherTopology matching branch i of the template topology
        """
        correspondingBranchIndices = []
        if templateTopology is None:
            templateTopology = self.templateTopology

        if otherTopology is None and self.extractedTopology is None:
            otherTopology = self.extractTopology()
        elif otherTopology is None and self.extractedTopology is not None:
            otherTopology = self.extractedTopology

        # get features vectors
        templateTopologyBranchFeatures = []
        otherTopologyBranchFeatures = []
        for branch in templateTopology.getBranches():
            templateTopologyBranchFeatures.append(
                self.getBranchFeatures(templateTopology, branch)
            )
        templateTopologyBranchFeatureVector = np.array(templateTopologyBranchFeatures)
        for branch in otherTopology.getBranches():
            otherTopologyBranchFeatures.append(
                self.getBranchFeatures(otherTopology, branch)
            )
        otherTopologyBranchFeatureVector = np.array(otherTopologyBranchFeatures)

        # normalization
        scaler = MinMaxScaler()
        templateTopologyBranchFeatureVectorNormalized = scaler.fit_transform(
            templateTopologyBranchFeatureVector
        )
        otherTopologyBranchFeatureVectorNormalized = scaler.fit_transform(
            otherTopologyBranchFeatureVector
        )

        # correspondance estimation
        correspondenceMatrix = distance_matrix(
            templateTopologyBranchFeatureVectorNormalized,
            otherTopologyBranchFeatureVectorNormalized,
        )
        _, branchCorrespondence = linear_sum_assignment(correspondenceMatrix)
        # for i, index in enumerate(templateBranchIndices):
        #     correspondingBranchPair = (index, otherBranchIndices[i])
        #     correspondingBranchIndices.append(correspondingBranchPair)
        return branchCorrespondence

    def calculateBranchCorresponanceAndLocalCoordinatsForPointSet(self, pointSet):
        """returns the branch correspondance and the corresponding local coordinates for each point of the given point set with respect to the templateTopology."""
        if self.extractedTopology is None:
            raise ValueError("No extracted topology. First extract a topology.")

        # calculate all necessary correspondances
        correspondanceMappingFromTemplateToExtractedBranches = (
            self.getCorrespondingBranches()
        )
        (
            correspondingBranchIndicesInExtractedTopology,
            localCoordinates,
        ) = self.extractedTopology.calculateBranchCorrespondanceAndLocalCoordinatesForPointSet(
            pointSet
        )

        # map from extracted corresponances to template corresponances
        correspondingBranchIndicesInTemplateTopology = []
        for branchIndex in correspondingBranchIndicesInExtractedTopology:
            correspondingBranchIndexInTemplateTopology = np.where(
                correspondanceMappingFromTemplateToExtractedBranches == branchIndex
            )[0][0]
            correspondingBranchIndicesInTemplateTopology.append(
                correspondingBranchIndexInTemplateTopology
            )
        return correspondingBranchIndicesInTemplateTopology, localCoordinates