import sys
import os
import numpy as np
from scipy.spatial import distance_matrix
from sklearn.neighbors import NearestNeighbors
import numbers
from warnings import warn

try:
    sys.path.append(os.getcwd().replace("/src/dimreduction/som", ""))
    from src.utils.utils import gaussian_kernel, knn
except:
    print("Imports for JSPR failed.")
    raise
try:
    sys.path.append(os.getcwd().replace("/src/dimreduction/l1median", ""))
    from src.dimreduction.dimensionalityReduction import DimensionalityReduction
except:
    print("Imports for L1-Median failed.")
    raise


class SelfOrganizingMap(DimensionalityReduction):
    """
    Implementation of the Self-Organizing-Map (SOM) algoritm according to the Papers:
    1) "Kohonen, T.: The self-organizing map: Proceedings of the IEEE, 37, 1990, 1464–1480" for the numNeighor-based method
    2) "Kohonen, T.: Essentials of the self-organizing map, Neural networks : the official journal of the International Neural Network Society, 78(9), 2023, 52–65"  for the kernel-based method

    Attributes:
    -------------
    Y: Jx3 np.ndarray
        input data points
    T: Ix3 np.ndarray
        neurons of the SOM
    alpha: float
        gain for updating the neurons
    alphaAnnealing: float
        annealing factor for the gain
    numNearestNeighbors: float
        number of nearest neighbors used to determine the neighborhood of a neuron
    numNearestNeighborsAnnealing: float
        annealing factor for the number of nearest neighbors
    sigma2: float
        factor for determining the neighborhood if the kernel method is used.
    sigma2Annealing : float
        annealing factor of sigma2
    kernelMethod: bool
        if the "original" numNeighbor-based implementation of the

    """

    def __init__(
        self,
        alpha=None,
        alphaAnnealing=None,
        numNearestNeighbors=None,
        numNearestNeighborsAnnealing=None,
        sigma2=None,
        sigma2Annealing=None,
        kernelMethod=False,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.numNearestNeighbors = (
            5 if numNearestNeighbors is None else numNearestNeighbors
        )
        self.numNearestNeighborsAnnealing = (
            0.9
            if numNearestNeighborsAnnealing is None
            else numNearestNeighborsAnnealing
        )
        self.alpha = 0.9 if alpha is None else alpha
        self.alphaAnnealing = 0.93 if alphaAnnealing is None else alphaAnnealing
        self.sigma2 = 1 if sigma2 is None else sigma2
        sigma2Annealing = 0.9 if sigma2Annealing is None else sigma2Annealing
        self.kernelMethod = kernelMethod

    def calculateReducedRepresentation(self):
        """
        Function to perform Self Organizing Map training (estimation of weights).
        """
        while self.iteration < self.max_iterations:
            # anneling of update parameter
            alpha = (self.alpha) ** (self.iteration) * self.alpha
            sigma2 = (self.sigma2) ** (self.iteration) * self.sigma2
            numNearestNeighbors = round(
                (self.numNearestNeighborsAnnealing) ** (self.iteration)
                * self.numNearestNeighbors
            )

            # find the winning neurons for the dataset
            distanceMatrix = distance_matrix(self.T, self.Y)
            if sigma2 <= np.finfo(float).eps:
                H = np.eye(self.N)
            else:
                H = alpha * gaussian_kernel(self.T, sigma2)
            winnerIdxs = np.argmin(distanceMatrix, axis=0)

            # update neurons
            if self.kernelMethod == True:
                YMean = np.zeros((self.N, self.D))
                NumCorrespondences = np.zeros(self.N)
                for j in range(0, self.N):
                    J = np.where(winnerIdxs == j)[0]
                    YMean[j, :] = np.sum(
                        self.Y[np.where(winnerIdxs == j)[0], :], axis=0
                    )
                    NumCorrespondences[j] = len(J)

                if np.any(NumCorrespondences == 0):
                    updateIdxs = np.where(NumCorrespondences != 0)[0]
                    for idx in updateIdxs:
                        YMeanWeighted = np.zeros(self.D)
                        NumCorrespondencesWeighted = 0
                        for j in range(0, self.N):
                            YMeanWeighted += H[idx, j] * YMean[j, :]
                            NumCorrespondencesWeighted += (
                                H[idx, j] * NumCorrespondences[j]
                            )
                        self.T[idx, :] = YMeanWeighted / NumCorrespondencesWeighted
                else:
                    self.T = (H @ YMean) / (H @ NumCorrespondences)[:, None]
            else:
                # determine neighbors
                (nearestNeighbors, _) = knn(self.T, self.T, numNearestNeighbors + 1)
                for i in range(self.N):
                    if np.any(winnerIdxs == i):
                        nearestNeighborIdxs = nearestNeighbors[i, :]
                        correspondingPointIdxs = np.where(winnerIdxs == i)[0]
                        yMean = np.sum(self.Y[correspondingPointIdxs, :], axis=0) / len(
                            correspondingPointIdxs
                        )
                        self.T[nearestNeighborIdxs, :] = self.T[
                            nearestNeighborIdxs, :
                        ] + alpha * (
                            np.tile(yMean, (len(nearestNeighborIdxs), 1))
                            - self.T[nearestNeighborIdxs, :]
                        )

            self.iteration += 1

            if callable(self.callback):
                self.callback()

        return self.T
