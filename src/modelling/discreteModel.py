import sys, os
from builtins import super
import numpy as np
import numbers
from warnings import warn

try:
    sys.path.append(os.getcwd().replace("/src/modelling", ""))
    from src.simulation.dlo import DeformableLinearObject
except:
    print("Imports for discrete Model failed.")
    raise


class finiteSegmentModel(object):
    """Implementation of a finite segment representation of a DLO.
    The implementation is based on the theoretic principles of Discrete Kirchoff Rods from the paper:
    "Miklos Bergou et al., Discrete Elastic Rods, ACM Transactions on Graphics (SIGGRAPH), 2008"
    and uses the Dynamics Animation and Robotics Toolkit (DART) to model the kinematics.

    Args:
        object (_type_): _description_
    """

    def __init__(
        self,
        L=None,
        numSegments=None,
        Rflex=None,
        Rtor=None,
        Roh=None,
        x0=None,
        gravity=None,
        *args,
        **kwargs
    ):

        if L is not None and (not isinstance(L, numbers.Number) or L < 0):
            raise ValueError(
                "Expected a positive float for length of the DLO instead got: {}".format(
                    L
                )
            )

        if numSegments is not None and (
            not isinstance(numSegments, numbers.Number) or numSegments < 1
        ):
            raise ValueError(
                "Expected a positive integer of at least 1 for the number of segments instead got: {}".format(
                    numSegments
                )
            )
        elif isinstance(numSegments, numbers.Number) and not isinstance(
            numSegments, int
        ):
            warn(
                "Received a non-integer value for number of segments: {}. Casting to integer.".format(
                    numSegments
                )
            )
            numSegments = int(numSegments)
        self.L = 1 if L is None else L
        self.numSegments = 10 if numSegments is None else numSegments
        self.Rflex = 1 if Rflex is None else Rflex
        self.Rtor = 1 if Rtor is None else Rtor
        self.Roh = 0.1 if Roh is None else Roh
        self.x0 = np.zeros(3) if x0 is None else x0
        self.gravity = np.array([0, 0, 9.81]) if gravity is None else gravity

        self.dlo = DeformableLinearObject(
            numSegments=numSegments, length=L, density=Roh, stiffness=Rflex
        )
