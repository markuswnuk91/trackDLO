import sys
import os
import numpy as np
from warnings import warn
from sklearn.neighbors import LocalOutlierFactor

try:
    sys.path.append(os.getcwd().replace("/src/localization/downsampling/random", ""))
    from src.localization.downsampling.downsampling import Downsampling
except:
    print("Imports for Random Sampler failed.")
    raise


class LocalOutlierFactorFilter(Downsampling):
    """
    Implements a random downsampling method
    """

    def __init__(self, numNeighbors, contamination, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.numNeighbors = numNeighbors
        self.contamination = contamination

    def sampleLOF(self, Y=None):
        """
        Function to perform lof outlier  filtering
        """
        if Y is not None:
            self.Y = Y
        self.lofFilter = LocalOutlierFactor(
            n_neighbors=self.numNeighbors,
            contamination=self.contamination,
        )
        self.filterResult = self.lofFilter.fit_predict(self.Y)
        self.negOutlierScore = self.lofFilter.negative_outlier_factor_
        self.T = self.Y[np.where(self.filterResult != -1)[0], :]
        self.Outliers = self.Y[np.where(self.filterResult != 1)[0], :]

        if callable(self.callback):
            self.callback()
        return self.T
