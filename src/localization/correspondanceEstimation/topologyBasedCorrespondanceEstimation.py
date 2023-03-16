import sys
import os
import numpy as np
import numbers
from warnings import warn

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

    def __init__(self, Y, numSeedPoints, templateTopology, *args, **kwargs):
        if type(Y) is not np.ndarray or Y.ndim != 2:
            raise ValueError("The source point cloud (Y) must be a 2D numpy array.")

        if Y.shape[0] < Y.shape[1]:
            raise ValueError(
                "The dimensionality is larger than the number of points. Possibly the wrong orientation of Y."
            )

        self.Y = Y
        self.templateTopology = templateTopology
        self.extractedTopology = self.extractTopology(numSeedPoints)

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

    def getExtractedTopology(self):
        return self.extractedTopology

    def getBranchFeatures(self, branch):
        """Determines a feature vector used for correspondance estimation between individual branches

        Args:
            branch (branch): branch for which the featues should be extracted

        Raises:
            NotImplementedError: _description_
        """
        featureList = []
        branchLength = branch.getLength()
        featureList.append(branchLength)
        branchFeatures = np.array(featureList)
        raise NotImplementedError

    def getCorrespondingBranches(self):
        """Determines the corresponding branches between the template topology and the extracted topology

        Returns:
        branchCorrespondances (list of tuples):
            list of tuples with indices of corresponding branches
        """
        templateTopologyFeatures = self.getTopologyFeatures(self.templateTopology)
        extractedTopologyFeatures = self.getTopologyFeatures(self.extractedTopology)

        raise NotImplementedError
