# tests dlo implementation
import os, sys
import dartpy as dart
import numpy as np
import math
import time as time
from pytest import approx

try:
    sys.path.append(os.getcwd().replace("/tests", ""))
    from src.modelling.finiteSegmentModel import FiniteSegmentModel
except:
    print("Imports for Finite Segment Model Tests failed.")
    raise


def test_FiniteSegmentModel():

    # test getCartesianPosition functions
    testDLO = FiniteSegmentModel(L=1.5, N=20)
    assert np.sum(testDLO.getCartesianPositionRootJoint() - np.zeros(3)) == approx(0)
    assert np.sum(
        testDLO.getCartesianPositionSegmentEnd(-1) - np.array([0, 0, 1.5])
    ) == approx(0)
    assert testDLO.getBodyNodeIndexFromLocalCoodinate(0) == 0
    assert testDLO.getBodyNodeIndexFromLocalCoodinate(0.01) == 0
    assert testDLO.getBodyNodeIndexFromLocalCoodinate(1) == 19
    assert testDLO.getCartesianPositionFromLocalCoordinate(0.1)[2] == approx(0.15)

    testDLO.convertPhiToBallJointPositions(0.1, np.array([0, 0, 1]))


if __name__ == "__main__":
    test_FiniteSegmentModel()
