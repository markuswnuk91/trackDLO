import sys
import os
import numpy as np
from warnings import warn
import random

try:
    sys.path.append(os.getcwd().replace("/src/localization/downsampling/random", ""))
    from src.localization.downsampling.downsampling import Downsampling
except:
    print("Imports for Random Sampler failed.")
    raise


class RandomSampling(Downsampling):
    """
    Implements a random downsampling method
    """

    def __init__(self, numSeedPoints, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.Y is not None:
            if numSeedPoints > len(self.Y):
                raise ValueError(
                    "Requested number of seed points larger than input point set. Expected  number of seed points to be lower than the size of the input sample."
                )
        self.numSeedPoints = numSeedPoints

    def sampleRandom(self, Y=None, numSeedPoints=None):
        """
        Function to perform random downsampling
        """
        if Y is not None:
            self.Y = Y
        if numSeedPoints is None:
            numSeedPoints = self.numSeedPoints
        random_indices = random.sample(range(0, len(self.Y)), numSeedPoints)
        self.T = self.Y[random_indices, :]
        if callable(self.callback):
            self.callback()
        return self.T
