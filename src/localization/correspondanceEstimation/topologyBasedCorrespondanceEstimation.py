import sys
import os
import numpy as np
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment

try:
    sys.path.append(
        os.getcwd().replace("/src/localization/correspondanceEstimation", "")
    )
    from src.localization.topologyExtraction.topologyExtraction import (
        TopologyExtraction,
    )
except:
    print("Imports for Topology-based Correspondance Estimation failed.")
    raise


class TopologyBasedCorrespondanceEstimation(TopologyExtraction):
    """Class to perform a topology based correspondance estimation between a given Template Topology of a BDLO and a topology extracted from a given point set

    Attributes:
    Y: numpy.array
        MxD array point cloud representation of the BDLO

    templateTopology: topologyModel
        topology model representing the topology of the BDLO
    """

    def __init__(self, templateTopology, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.templateTopology = templateTopology

    def getExtractedTopology(self):
        if self.extractedTopology is None:
            self.extractTopology()
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
            pass
        return self.extractedTopology

    def getBranchFeatures(self, branch):
        """Determines a feature vector used for correspondance estimation between individual branches

        Args:
            branch (branch): branch for which the featues should be extracted

        Raises:
            NotImplementedError: _description_
        """
        featureList = []
        branchLength = branch.getBranchInfo()["length"]
        featureList.append(branchLength)
        branchFeatures = np.array(featureList)
        return branchFeatures

    def getCorrespondingBranches(self, templateTopology=None, otherTopology=None):
        """Determines the corresponding branches between the template topology and an other topology

        Returns:
        branchCorrespondances (list of tuples):
            list of tuples with indices of corresponding branches
        """
        correspondingBranchIndices = []
        if templateTopology is None:
            templateTopology = self.templateTopology

        if otherTopology is None and self.extractedTopology is None:
            otherTopology = self.getExtractedTopology()
        elif otherTopology is None and self.extractedTopology is not None:
            otherTopology = self.extractedTopology

        templateTopologyBranchFeatures = []
        otherTopologyBranchFeatures = []
        for branch in templateTopology.getBranches():
            templateTopologyBranchFeatures.append(self.getBranchFeatures(branch))
        templateTopologyBranchFeatureVector = np.array(templateTopologyBranchFeatures)
        for branch in otherTopology.getBranches():
            otherTopologyBranchFeatures.append(self.getBranchFeatures(branch))
        otherTopologyBranchFeatureVector = np.array(otherTopologyBranchFeatures)

        correspondenceMatrix = distance_matrix(
            templateTopologyBranchFeatureVector, otherTopologyBranchFeatureVector
        )
        templateBranchIndices, otherBranchIndices = linear_sum_assignment(
            correspondenceMatrix
        )
        for i, index in enumerate(templateBranchIndices):
            correspondingBranchPair = (index, otherBranchIndices[i])
            correspondingBranchIndices.append(correspondingBranchPair)
        return correspondingBranchIndices
