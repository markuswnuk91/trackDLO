rom builtins import super
import os
import sys
import numpy as np
import numbers
from warnings import warn
from scipy.optimize import least_squares
import json

try:
    sys.path.append(os.getcwd().replace("/src/reconstruction/discrete", ""))
    from src.reconstruction.shapeReconstruction import ShapeReconstruction
    from src.simulation.dlo import DeformableLinearObject
except:
    print("Imports for discrete shape reconstruction failed.")
    raise

class DiscreteReconstruction(ShapeReconstruction):

    def __init__(
        self,
        wPosDiff=None,
        annealingFlex=None,
        annealingTor=None,
        *args,
        **kwargs
    ):
        self.annealingFlex = 1 if annealingFlex is None else annealingFlex
        self.annealingTor = 1 if annealingTor is None else annealingTor
        self.iter = 0

    def updateParameters(self, optimVars):
        # update DLO
        # update self.X
        self.iter += 1
        raise NotImplementedError

    def costFun(self, optimVars):
        error = (
                self.annealingFlex**self.iter * self.evalUflex(self.L, self.numSc)
                + self.annealingTor**self.iter * self.evalUtor(self.L, self.numSc)
                + self.evalUgrav(self.L, self.numSc)
                + self.wPosDiff * np.square(np.linalg.norm(self.Y - self.X))
                # + self.wPosDiff * (1 - np.exp(-np.linalg.norm(self.Y - self.X) / 1000))
                # + np.sum(self.aPsi)
            )