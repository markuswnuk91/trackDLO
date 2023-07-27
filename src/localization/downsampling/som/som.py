import sys
import os
import numpy as np
from scipy.spatial import distance_matrix
from sklearn.neighbors import NearestNeighbors
import numbers
from warnings import warn
import time

try:
    sys.path.append(os.getcwd().replace("/src/localization/downsampling/som", ""))
    from src.utils.utils import gaussian_kernel, knn
    from src.localization.downsampling.datareduction import (
        DataReduction,
    )
except:
    print("Imports for SOM failed.")
    raise


class SelfOrganizingMap(DataReduction):
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
    method: string
        kernel, knn, legacy, what method should be used to perfrom the SOM updates
    """

    def __init__(
        self,
        alpha=None,
        alphaAnnealing=None,
        numNearestNeighbors=None,
        numNearestNeighborsAnnealing=None,
        minNumNearestNeighbors=None,
        sigma2=None,
        sigma2Annealing=None,
        sigma2Min=None,
        method="kernel",
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
        self.alphaAnnealing = 1 if alphaAnnealing is None else alphaAnnealing
        self.sigma2 = 1 if sigma2 is None else sigma2
        self.sigma2Annealing = 0.9 if sigma2Annealing is None else sigma2Annealing
        self.method = method
        self.minNumNearestNeighbors = (
            0 if minNumNearestNeighbors is None else minNumNearestNeighbors
        )
        self.sigma2Min = np.finfo(float).eps if sigma2Min is None else sigma2Min
        self.runTimes = {}
        self.runTimes["perIteration"] = []

    def calculateReducedRepresentation(self, Y=None, X=None):
        """
        Function to perform Self Organizing Map training (estimation of weights).
        """
        runTime_start = time.time()
        if Y is not None:
            self.Y = Y
            (self.M, _) = self.Y.shape
        if X is not None:
            self.X = X
        else:
            self.X = self.sampleRandom(self.Y, self.numSeedPoints)
        (self.N, self.D) = self.X.shape
        self.T = self.X

        while self.iteration < self.max_iterations:
            runtimePerInteration_start = time.time()
            # anneling of update parameter
            alpha = (self.alphaAnnealing) ** (self.iteration) * self.alpha
            # alpha = self.alpha * (1 - self.iteration / self.max_iterations)
            sigma2 = (self.sigma2Annealing) ** (self.iteration) * self.sigma2
            if sigma2 < self.sigma2Min:
                sigma2 = self.sigma2Min
            numNearestNeighbors = round(
                self.numNearestNeighbors * (1 - self.iteration / self.max_iterations)
            )
            if numNearestNeighbors < self.minNumNearestNeighbors:
                numNearestNeighbors = self.minNumNearestNeighbors

            # update neurons
            if self.method == "kernel":
                # find the winning neurons for the dataset
                distanceMatrix = distance_matrix(self.T, self.Y)
                if sigma2 <= np.finfo(float).eps:
                    H = np.eye(self.N)
                else:
                    H = alpha * gaussian_kernel(self.T, np.square(sigma2))
                    H = H - np.diag(np.diag(H)) + np.eye(len(H))
                winnerIdxs = np.argmin(distanceMatrix, axis=0)
                nearestPointIdxs = np.argmin(distanceMatrix, axis=1)
                YMean = np.zeros((self.N, self.D))
                NumCorrespondences = np.zeros(self.N)
                for j in range(0, self.N):
                    J = np.where(winnerIdxs == j)[0]
                    # YMean[j, :] = self.T[j, :] + np.sum(
                    #     self.Y[np.where(winnerIdxs == j)[0], :], axis=0
                    # )
                    YMean[j, :] = np.mean(
                        np.vstack(
                            (
                                self.Y[nearestPointIdxs[j]],
                                self.Y[np.where(winnerIdxs == j)[0], :],
                            ),
                        ),
                        axis=0,
                    )
                    NumCorrespondences[j] = len(J) + 1

                    # if np.any(NumCorrespondences == 0):
                    # updateIdxs = np.where(NumCorrespondences != 0)[0]
                    # for idx in updateIdxs:
                    #     YMeanWeighted = np.zeros(self.D)
                    #     NumCorrespondencesWeighted = 0
                    #     for j in range(0, self.N):
                    #         YMeanWeighted += H[idx, j] * YMean[j, :]
                    #         NumCorrespondencesWeighted += (
                    #             H[idx, j] * NumCorrespondences[j]
                    #         )
                    #     self.T[idx, :] = YMeanWeighted / NumCorrespondencesWeighted
                    # else:
                # self.T = (H @ YMean) / (H @ NumCorrespondences)[:, None]

                for i in range(0, self.N):
                    sum_nj_hji = 0
                    sum_nj_hji_xmj = np.zeros(self.D)
                    for j in range(0, self.N):
                        sum_nj_hji_xmj += NumCorrespondences[j] * H[j, i] * YMean[j]
                        sum_nj_hji += NumCorrespondences[j] * H[j, i]
                    self.T[i, :] = sum_nj_hji_xmj / sum_nj_hji
            elif self.method == "knn":
                for i in range(0, self.N):
                    # find the winning neurons for the dataset
                    distanceMatrix = distance_matrix(self.T, self.Y)
                    winnerIdxs = np.argmin(distanceMatrix, axis=0)
                    # determine neighbors
                    # (nearestNeighbors, _) = knn(self.T, self.T, numNearestNeighbors + 1)
                    knn = NearestNeighbors(n_neighbors=(numNearestNeighbors + 1))
                    knn.fit(self.T)
                    (neighborDistances, nearestNeighbors) = knn.kneighbors(
                        self.T, return_distance=True
                    )
                    if np.any(winnerIdxs == i):
                        idxsInNeighborArray = np.where(nearestNeighbors[i, :] != i)[0]
                        nearestNeighborIdxs = nearestNeighbors[i, idxsInNeighborArray]
                        nearestNeighborDistances = neighborDistances[
                            i, idxsInNeighborArray
                        ]
                        correspondingPointIdxs = np.where(winnerIdxs == i)[0]
                        yMean = np.sum(self.Y[correspondingPointIdxs, :], axis=0) / len(
                            correspondingPointIdxs
                        )
                        self.T[i, :] = self.T[i, :] + alpha * (yMean - self.T[i, :])
                        if len(nearestNeighborIdxs) > 0 and numNearestNeighbors >= 1:
                            for k, neighborIdx in enumerate(nearestNeighborIdxs):
                                self.T[neighborIdx, :] = self.T[neighborIdx, :] + (
                                    alpha * (yMean - self.T[neighborIdx, :])
                                )
                        # nearestNeighborIdxs = np.insert(nearestNeighborIdxs, 0, i)
                        # self.T[nearestNeighborIdxs, :] = self.T[
                        #     nearestNeighborIdxs, :
                        # ] + alpha * (
                        #     np.tile(yMean, (len(nearestNeighborIdxs), 1))
                        #     - self.T[nearestNeighborIdxs, :]
                        # )
            else:
                for i, y in enumerate(self.Y):
                    # find the winning neurons for the datapoint
                    distances = distance_matrix(self.T, np.expand_dims(y, 0))
                    winnerIdx = np.argmin(distances, axis=0)[0]

                    # determine neighborhood
                    # knn = NearestNeighbors(n_neighbors=(numNearestNeighbors + 1))
                    # knn.fit(self.T)
                    # (neighborDistances, nearestNeighbors) = knn.kneighbors(
                    #     self.T, return_distance=True
                    # )
                    # idxsInNeighborArray = np.where(
                    #     nearestNeighbors[winnerIdx, :] != winnerIdx
                    # )[0]
                    # nearestNeighborIdxs = nearestNeighbors[
                    #     winnerIdx, idxsInNeighborArray
                    # ]

                    if winnerIdx - round(numNearestNeighbors / 2) < 0:
                        nearestNeighborIdxs = np.arange(round(numNearestNeighbors))
                        nearestNeighborIdxs = np.setdiff1d(
                            nearestNeighborIdxs, winnerIdx
                        )
                    elif winnerIdx + round(numNearestNeighbors / 2) >= self.N:
                        nearestNeighborIdxs = np.arange(
                            self.N - round(numNearestNeighbors / 2), self.N
                        )
                        nearestNeighborIdxs = np.setdiff1d(
                            nearestNeighborIdxs, winnerIdx
                        )
                    else:
                        nearestNeighborIdxs = np.arange(
                            winnerIdx - round(numNearestNeighbors / 2),
                            winnerIdx + round(numNearestNeighbors / 2) + 1,
                        )
                        nearestNeighborIdxs = np.setdiff1d(
                            nearestNeighborIdxs, winnerIdx
                        )

                    # update winner neuron
                    self.T[winnerIdx, :] = self.T[winnerIdx, :] + alpha * (
                        y - self.T[winnerIdx, :]
                    )
                    for neighborIdx in nearestNeighborIdxs:
                        # update neighboring neurons
                        self.T[neighborIdx, :] = self.T[neighborIdx, :] + alpha * (
                            y - self.T[neighborIdx, :]
                        )

            self.iteration += 1
            runtimePerInteration_end = time.time()

            self.runTimes["perIteration"].append(
                runtimePerInteration_end - runtimePerInteration_start
            )
            if callable(self.callback):
                self.callback()

        runTime_end = time.time()
        self.runTimes["withVisualization"] = runTime_end - runTime_start
        self.runTimes["withoutVisualization"] = np.sum(self.runTimes["perIteration"])
        return self.T
