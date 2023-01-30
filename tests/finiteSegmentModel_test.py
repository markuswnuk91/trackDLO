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

    # testing angle conversion
    qReference = np.zeros(testDLO.skel.getNumDofs())
    qReference[0] = np.pi / 4
    qReference[1] = np.pi / 3
    qReference[2] = np.pi / 2
    qReference[4] = 1
    testDLO.skel.setPositions(qReference)
    testDLO.mapDartPositionsToAngles(qReference)
    (
        x0_test,
        rot0_test,
        alphas_test,
        betas_test,
        gammas_test,
    ) = testDLO.mapDartPositionsToAngles(qReference)
    qTest = testDLO.mapAnglesToDartPositions(
        x0_test,
        rot0_test,
        alphas_test,
        betas_test,
        gammas_test,
    )
    assert np.sum(qTest - qReference) == approx(0)

    # testing angle conversion
    alphas_test = np.zeros(int((testDLO.skel.getNumDofs() - 6) / 3))
    betas_test = np.zeros(int((testDLO.skel.getNumDofs() - 6) / 3))
    gammas_test = np.zeros(int((testDLO.skel.getNumDofs() - 6) / 3))
    x0_test = np.zeros(3)
    rot0_test = np.zeros(3)

    alphas_test[0] = np.pi / 2
    betas_test[0] = np.pi / 2

    qTest = testDLO.mapAnglesToDartPositions(
        x0_test,
        rot0_test,
        alphas_test,
        betas_test,
        gammas_test,
    )


if __name__ == "__main__":
    test_FiniteSegmentModel()
