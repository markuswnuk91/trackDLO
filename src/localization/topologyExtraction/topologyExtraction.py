import sys
import os
import numpy as np
import numbers
from warnings import warn
from scipy.sparse.csgraph import minimum_spanning_tree, shortest_path
from scipy.spatial import distance_matrix

try:
    sys.path.append(os.getcwd().replace("/src/localization/topologyExtraction", ""))
    from src.localization.downsampling.som.som import SelfOrganizingMap
    from src.localization.downsampling.l1median.l1Median import L1Median
    from src.localization.downsampling.random.randomSampling import RandomSampling
    from src.localization.downsampling.filter.lofFilter import LocalOutlierFactorFilter
    from src.localization.topologyExtraction.minimalSpanningTreeTopology import (
        MinimalSpanningTreeTopology,
    )
except:
    print("Imports for Topology Extraction failed.")
    raise


class TopologyExtraction(object):
    """This class implements a number of filtering, dimensionality and data reduction methods to extract a BDLO topology from a given point set.

    Attributes:
    Y: numpy array
        MxD array of unordered, unstructured data points from which the topology should be extracted.

    X: numpy array
        NxD array of unordered but well aligned points for the graph reconstruction.

    N: int
        Number of source points

    M: int
        Number of data points

    D: int
        Dimensionality of source and data points

    appliedReductionMethods: list
        list of reduction methods applied by the topology extraction

    appliedFilterMethods: list
        list of filter methods applied by the topology extraction
    """

    def __init__(
        self,
        Y,
        numSeedPoints,
        randomSamplingParameters=None,
        somParameters=None,
        l1Parameters=None,
        lofOutlierFilterParameters=None,
        *args,
        **kwargs
    ):

        if type(Y) is not np.ndarray or Y.ndim != 2:
            raise ValueError("The source point cloud (Y) must be a 2D numpy array.")

        if Y.shape[0] < Y.shape[1]:
            raise ValueError(
                "The dimensionality is larger than the number of points. Possibly the wrong orientation of Y."
            )
        if type(numSeedPoints) is not int:
            warn("Obtained non integer for number of seed points. Casting to integer.")
            numseedPoints = int(numSeedPoints)
        elif numSeedPoints > Y.shape[0]:
            raise ValueError(
                "Number of seed points larger than number of source points. Use less seedpoints."
            )
        elif numSeedPoints <= 0:
            raise ValueError("Too few seed points.")
        self.Y = Y
        self.numSeedPoints = numSeedPoints
        self.topology = None

        if randomSamplingParameters is None:
            self.randomSamplingParameters = {"numSeedPoints": int(len(Y) / 3)}
        else:
            self.randomSamplingParameters = randomSamplingParameters

        if somParameters is None:
            self.somParameters = {
                "alpha": 1,
                "numNearestNeighbors": 30,
                "numNearestNeighborsAnnealing": 0.8,
                "sigma2": 0.03,
                "alphaAnnealing": 0.9,
                "sigma2Annealing": 0.8,
                "kernelMethod": False,
                "max_iterations": 30,
            }
        else:
            self.somParameters = somParameters
        if l1Parameters is None:
            self.l1Parameters = {
                "h": 0.12,
                "hAnnealing": 0.9,
                "hReductionFactor": 1,
                "mu": 0.35,
                "max_iterations": 30,
            }
        else:
            self.l1Parameters = l1Parameters
        if lofOutlierFilterParameters is None:
            self.lofOutlierFilterParameters = {
                "numNeighbors": 15,
                "contamination": 0.1,
            }
        else:
            self.lofOutlierFilterParameters = lofOutlierFilterParameters

        self.randomSampling = RandomSampling(**self.randomSamplingParameters)
        self.selfOrganizingMap = SelfOrganizingMap(Y=self.Y, **self.somParameters)
        self.l1Median = L1Median(Y=self.Y, **self.l1Parameters)
        self.lofOutlierFilter = LocalOutlierFactorFilter(
            **self.lofOutlierFilterParameters
        )

    def extractTopology(
        self, numSeedPoints=None, extractionMethod="random_som_l1_outlier"
    ):
        if numSeedPoints is None:
            numSeedPoints = self.numseedPoints
        if extractionMethod == "random_som_l1_outlier":
            seedPoints = self.samplePointsRandom(self.Y, numSeedPoints)
            reducedPointSet = self.reducePointSet_SOM(self.Y, seedPoints)
            reducedPointSet = self.reducePointSet_L1(self.Y, reducedPointSet)
            filteredPointSet = self.filterOutliers(reducedPointSet, filterMethod="lof")
            self.topology = MinimalSpanningTreeTopology(filteredPointSet)
        else:
            raise NotImplementedError

        return self.topology

    def reducePointSet_SOM(
        self,
        pointSet,
        seedPoints,
    ):
        """downsamples a pointSet with the self organizing map

        Args:
            pointSet (np.array): MxD array of source points
            seedPoints (np.array): NxD array of seedpoints

        Returns:
            reducedPoints(np.array), reductionMethod: returns the reduced point set
        """
        reducedPoints = self.selfOrganizingMap.calculateReducedRepresentation(
            pointSet, seedPoints
        )
        return reducedPoints

    def reducePointSet_L1(
        self,
        pointSet,
        seedPoints,
    ):
        """downsamples a pointSet with the l1-median

        Args:
            pointSet (np.array): MxD array of source points
            seedPoints (np.array): NxD array of seedpoints

        Returns:
            reducedPoints(np.array), reductionMethod: returns the reduced point set
        """
        reducedPoints = self.l1Median.calculateReducedRepresentation(
            pointSet, seedPoints
        )
        return reducedPoints

        # def reducePointSet(
        #     self,
        #     pointSet,
        #     seedPoints,
        #     reductionMethod: str = "som",
        #     reductionParameters: dict = None,
        # ):
        #     """downsamples a pointSet with the specified method

        #     Args:
        #         pointSet (np.array): MxD array of source points
        #         seedPoints (np.array): NxD array of seedpoints
        #         reductionMethod (str): "som" or "l1" specifies the reduction method, defaults to "som".
        #         reductionParameters (dict): set parameters for the chosen reduction method. If not specified, default parameters are used.
        #         callback (function): callable function that is called once after the reduction is finished.

        #     Raises:
        #         NotImplementedError: for other methods than "som" and "l1".

        #     Returns:
        #         reducedPoints(np.array), reductionMethod: returns the reduced point set, and the reduction method
        #     """
        #     if reductionMethod == "som":
        #         reductionMetho
        #         if reductionParameters is None:
        #             reductionParameters = self.somParameters
        #         reductionParameters["Y"] = pointSet
        #         reductionParameters["X"] = seedPoints
        #         reductionAlgorithm = SelfOrganizingMap(**reductionParameters)
        #     elif reductionMethod == "l1":
        #         if reductionParameters is None:
        #             reductionParameters = self.l1Parameters
        #         reductionParameters["Y"] = pointSet
        #         reductionParameters["X"] = seedPoints
        #         reductionAlgorithm = L1Median(**reductionParameters)
        #     else:
        #         raise NotImplementedError

        # reducedPoints = reductionAlgorithm.calculateReducedRepresentation()
        # self.appliedReductionMethods.append(reductionAlgorithm)
        # return reducedPoints

    def filterOutliers(self, pointSet, filterMethod="lof"):
        """filter a point set for outliers with the specified method

        Args:
            pointSet (np.array): array of point to be filtered
            filterMethod (str, optional): filter method. Defaults to "lof".
            filterParams (dict, optional): parameters for the filter method. If no parameters are provided the default values are used.
            callback (function, optional): callable function that is called once after the reduction is finished.

        Raises:
            NotImplementedError: for other methods than "lof"

        Returns:
            filteredPointset(np.array): filtered point set
        """
        if filterMethod == "lof":
            filteredPointSet = self.lofOutlierFilter.calculateReducedRepresentation(
                pointSet
            )
        else:
            raise NotImplementedError
        return filteredPointSet

    def samplePointsRandom(self, pointSet, numSeedPoints):
        self.randomSampling.numSeedPoints = numSeedPoints
        self.randomSampling.calculateReducedRepresentation(pointSet)
        return self.randomSampling.T
