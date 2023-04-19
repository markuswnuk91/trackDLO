# tests dart installation
import os, sys
import dartpy as dart
import numpy as np
import math
import time as time
from pytest import approx
from unittest import TestCase
import numbers

try:
    sys.path.append(os.getcwd().replace("/tests", ""))
    from src.simulation.deformationJacobian import linearDeformationJacobian
    from src.simulation.dlo import DeformableLinearObject
except:
    print("Imports for BDLO testing failed.")
    raise


def test_deformationJacobian():
    testDLO = DeformableLinearObject(10)
    q = testDLO.getPositions()
    vGripper = np.array(
        [
            0,
            1,
            0,
        ]
    )
    res = linearDeformationJacobian(
        vGripper=vGripper,
        skel=testDLO.skel.clone(),
        graspNodeIdx=5,
        q=q,
        k=1,
        d=0.01,
    )

    print(res[0])


# def calculateGeodesicDistance(skel, bodyNodeStart,bodyNodeEnd):
#     for idx in range(startIdx, endIdx):
#         skel.getBodyNode(idx).getParentJoint().getTransformToChildBodyNode()

if __name__ == "__main__":
    test_deformationJacobian()
