import os
import sys
import numpy as np
from numpy import genfromtxt
from pytest import approx

try:
    sys.path.append(os.getcwd().replace("/tests", ""))
    from src.tracking.utils.utils import knn
except:
    print("Imports for CPD failed.")
    raise


def testKNN():
    (J_np, dist) = knn(X, X, k)
    J_refMatlab = genfromtxt(
        "tests/testdata/spr/knn_J.csv", delimiter=","
    )  # index array from matalab
    D_refMatlab = genfromtxt(
        "tests/testdata/spr/knn_D.csv", delimiter=","
    )  # index array from matalab
    J_refMatlab = J_refMatlab - 1  # subract 1 to convert to python indices
    # sort rows to be comparable
    J_refMatlab = np.sort(J_refMatlab[:, :k], 1)
    D_refMatlab[
        D_refMatlab <= np.finfo(float).eps
    ] = 0  # makes sure we do not get negative values (due to small negative numbers in D_refMatlab under machine precision)
    D_refMatlab = np.sqrt(D_refMatlab[:, :k])

    J_np = np.sort(J_np, 1)
    for i in range(J_np.shape[0]):
        assert np.array_equal(J_np[i, :], J_refMatlab[i, :]) or np.sum(
            D_refMatlab[i, :]
        ) - np.sum(dist[i, :]) == approx(0, abs=1e-5)
    assert J_np.shape == (X.shape[0], k)


X = genfromtxt("tests/testdata/spr/Xinit.csv", delimiter=",")
Y = genfromtxt("tests/testdata/spr/Y.csv", delimiter=",")
k = 7
testKNN()
