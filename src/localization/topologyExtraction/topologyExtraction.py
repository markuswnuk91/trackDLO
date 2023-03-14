import sys
import os
import numpy as np
import numbers
from warnings import warn
from scipy.sparse.csgraph import minimum_spanning_tree, shortest_path
from scipy.spatial import distance_matrix
from sklearn.neighbors import LocalOutlierFactor

try:
    sys.path.append(os.getcwd().replace("/src/localization/topologyExtraction", ""))
    from src.dimreduction.som.som import SelfOrganizingMap
    from src.dimreduction.l1median.l1Median import L1Median
except:
    print("Imports for Topology Extraction failed.")
    raise


class TopologyExtraction(object):
    """

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
    """

    def __init__(
        self,
        Y,
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
        self.Y = Y
        self.topology = None
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

    def extractTopology():
        """
        Placeholder for child classes.
        """
        raise NotImplementedError(
            "Method to extract the topology should be defined in child classes."
        )

    def reducePointSet(
        self,
        pointSet,
        seedPoints,
        reductionMethod: str = "som",
        reductionParameters: dict = None,
        callback=None,
    ):
        """downsamples a pointSet with the specified method

        Args:
            pointSet (np.array): MxD array of source points
            seedPoints (np.array): NxD array of seedpoints
            reductionMethod (str): "som" or "l1" specifies the reduction method, defaults to "som".
            reductionParameters (dict): set parameters for the chosen reduction method. If not specified, default parameters are used.
            callback (function): callable function that is called once after the reduction is finished.

        Raises:
            NotImplementedError: _description_

        Returns:
            reducedPoints(np.array), reductionMethod: returns the reduced point set, and the reduction method
        """
        if reductionMethod == "som":
            if reductionParameters is None:
                reductionParameters = self.somParameters
            reductionParameters["Y"] = pointSet
            reductionParameters["X"] = seedPoints
            myReduction = SelfOrganizingMap(**reductionParameters)
        elif reductionMethod == "l1":
            if reductionParameters is None:
                reductionParameters = self.l1Parameters
            reductionParameters["Y"] = pointSet
            reductionParameters["X"] = seedPoints
            myReduction = L1Median(**reductionParameters)
        else:
            raise NotImplementedError

        if callable(callback):
            myReduction.register(callback)

        reducedPoints = myReduction.calculateReducedRepresentation()
        return reducedPoints, myReduction

    def filterOutliers(
        pointSet, filterMethod="lof", filterParams: dict = None, callback=None
    ):
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
            if filterParams is None:
                filterParams = self.lofOutlierFilterParameters
            lofFilter = LocalOutlierFactor(
                n_neighbors=filterParams["numNeighbors"],
                contamination=filterParams["contamination"],
            )
            filterResult = lofFilter.fit_predict(pointSet)
            negOutlierScore = lofFilter.negative_outlier_factor_
            filteredPointSet = pointSet[np.where(filterResult != -1)[0], :]
        else:
            raise NotImplementedError

        if callable(callback):
            callback()
        return filteredPointSet
