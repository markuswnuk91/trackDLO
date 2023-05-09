import sys
import os
import numpy as np
import numbers
from warnings import warn
from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import shortest_path
from sklearn import preprocessing

try:
    sys.path.append(os.getcwd().replace("/src/localization/topologyExtraction", ""))
    from src.localization.downsampling.som.som import SelfOrganizingMap
    from src.localization.downsampling.l1median.l1Median import L1Median
    from src.localization.downsampling.random.randomSampling import RandomSampling
    from src.localization.downsampling.filter.lofFilter import LocalOutlierFactorFilter
    from src.localization.topologyExtraction.minimalSpanningTreeTopology import (
        MinimalSpanningTreeTopology,
    )
    from src.utils.utils import minimalSpanningTree
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
        # if type(numSeedPoints) is not int:
        #     warn("Obtained non integer for number of seed points. Casting to integer.")
        #     numseedPoints = int(numSeedPoints)
        # elif numSeedPoints > Y.shape[0]:
        #     raise ValueError(
        #         "Number of seed points larger than number of source points. Use less seedpoints."
        #     )
        # elif numSeedPoints <= 0:
        #     raise ValueError("Too few seed points.")
        self.Y = Y
        self.X = self.Y
        self.reducedPointSetsRandom = []
        self.reducedPointSetsSOM = []
        self.reducedPointSetsL1 = []
        self.reducedPointSetsFilter = []
        self.extractedTopology = None
        self.extractedFeatureMatrix = None

        if somParameters is None:
            self.somParameters = {
                "numSeedPoints": 100,
                "alpha": 1,
                "alphaAnnealing": 0.93,
                "sigma2": 0.1,
                "sigma2Min": 0.01,
                "sigma2Annealing": 0.9,
                "method": "kernel",
                "max_iterations": 30,
            }
        else:
            self.somParameters = somParameters
        if l1Parameters is None:
            self.l1Parameters = {
                "numSeedPoints": 300,
                "h": 0.03,
                "hAnnealing": 0.9,
                "hMin": 0.01,
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
        if randomSamplingParameters is None:
            self.randomSamplingParameters = {"numSeedPoints": int(len(Y) / 3)}
        else:
            self.randomSamplingParameters = randomSamplingParameters
        self.randomSampling = RandomSampling(**self.randomSamplingParameters)
        self.selfOrganizingMap = SelfOrganizingMap(Y=self.Y, **self.somParameters)
        self.l1Median = L1Median(Y=self.Y, **self.l1Parameters)
        self.lofOutlierFilter = LocalOutlierFactorFilter(
            **self.lofOutlierFilterParameters
        )

    def runTopologyExtraction(
        self, numSeedPoints, extractionMethod="random_som_l1_outlier"
    ):
        if extractionMethod == "random_som_l1_outlier":
            seedPoints = self.randomSample(self.Y, numSeedPoints)
            reducedPointSet = self.reducePointSetSOM(self.Y, seedPoints)
            reducedPointSet = self.reducePointSetL1(self.Y, reducedPointSet)
            filteredPointSet = self.filterOutliers(reducedPointSet, filterMethod="lof")
            self.extractedTopology = MinimalSpanningTreeTopology(filteredPointSet)
            self.X = filteredPointSet
        else:
            raise NotImplementedError
        return self.extractedTopology

    def determineGeodesicDistance(self, reducedPointSet, densePointSet):
        combinedPointSet = np.vstack((reducedPointSet, densePointSet))
        AdjMinSpan = minimalSpanningTree(
            distance_matrix(combinedPointSet, combinedPointSet)
        )
        (pathDistanceMatrix, _) = shortest_path(
            AdjMinSpan,
            method="auto",
            directed=False,
            return_predecessors=True,
            unweighted=False,
            overwrite=False,
            indices=list(range(0, len(reducedPointSet))),
        )
        pathDistanceMatrix = pathDistanceMatrix[:, : len(reducedPointSet)]
        return pathDistanceMatrix

    def extractTopology(self, reducedPointSet, densePointSet=None, method="combined"):
        if method == "geodesic":
            if densePointSet is None:
                densePointSet = self.Y
            cartesianDistanceMatrix = distance_matrix(reducedPointSet, reducedPointSet)
            geodesicDistanceMatrix = self.determineGeodesicDistance(
                reducedPointSet, densePointSet
            )
            AdjGeodesicMinSpan = minimalSpanningTree(geodesicDistanceMatrix)
            connectivityGeodesicMinSpan = np.array(AdjGeodesicMinSpan != 0, dtype=int)
            self.extractedFeatureMatrix = (
                connectivityGeodesicMinSpan * cartesianDistanceMatrix
            )
        elif method == "cartesian":
            self.extactedFeatureMatrix = distance_matrix(
                reducedPointSet, reducedPointSet
            )
        elif method == "combined":
            if densePointSet is None:
                densePointSet = self.Y
            cartesianDistanceMatrix = distance_matrix(reducedPointSet, reducedPointSet)
            geodesicDistanceMatrix = self.determineGeodesicDistance(
                reducedPointSet, densePointSet
            )
            # normalization
            geodesicDistanceMatrixNormalized = (
                preprocessing.MinMaxScaler().fit_transform(geodesicDistanceMatrix)
            )
            # cartesianDistanceMatrixNormalized = (
            #     preprocessing.MinMaxScaler().fit_transform(cartesianDistanceMatrix)
            # )
            # combinedFeatureMatrix = (
            #     geodesicDistanceMatrixNormalized * cartesianDistanceMatrixNormalized
            # )
            self.extactedFeatureMatrix = (
                geodesicDistanceMatrixNormalized * cartesianDistanceMatrix
            )
        else:
            raise NotImplementedError
        extractedTopology = MinimalSpanningTreeTopology(
            **{
                "X": reducedPointSet,
                "featureMatrix": self.extactedFeatureMatrix,
            },
        )
        self.extractedTopology = extractedTopology
        return extractedTopology

    def reducePointSetSOM(
        self,
        pointSet,
        seedPoints=None,
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
        self.reducedPointSetsSOM.append(reducedPoints)
        return reducedPoints

    def reducePointSetL1(
        self,
        pointSet,
        seedPoints=None,
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
        self.reducedPointSetsL1.append(reducedPoints)
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
            reducedPoints = self.lofOutlierFilter.sampleLOF(pointSet)
        else:
            raise NotImplementedError
        self.reducedPointSetsSOM.append(reducedPoints)
        return reducedPoints

    def randomSample(self, pointSet, numSeedPoints):
        self.randomSampling.numSeedPoints = numSeedPoints
        reducedPointSet = self.randomSampling.sampleRandom(pointSet)
        self.X = reducedPointSet
        self.reducedPointSetsRandom.append(reducedPointSet)
        return reducedPointSet
