import os
import sys
import argparse
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
from pytest import approx

try:
    sys.path.append(os.getcwd().replace("/tests", ""))
    from src.tracking.spr.spr import StructurePreservedRegistration
except:
    print("Imports for SPR failed.")
    raise
vis = True  # enable for visualization


def difference_Matrix(X, Y):
    """
    Calculate gaussian kernel matrix.
    Attributes
    ----------
    X: numpy array
        NxD array of points

    Y: numpy
        MxD array of points

    Returns
    -------
    K: numpy array
            NxM array of differences
    """
    K = X[:, None, :] - Y[None, :, :]
    K = np.square(K)
    K = np.sum(K, 2)
    return K


def visualize(iteration, error, X, Y, ax):
    plt.cla()
    ax.scatter(X[:, 0], X[:, 1], color="red", label="Target")
    ax.scatter(Y[:, 0], Y[:, 1], color="blue", label="Source")
    plt.text(
        0.87,
        0.92,
        "Iteration: {:d}, error{}".format(iteration, error),
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax.transAxes,
        fontsize="x-large",
    )
    ax.legend(loc="upper left", fontsize="x-large")
    plt.draw()
    plt.pause(0.001)


def runSPR():
    X = np.loadtxt("tests/testdata/pycpd/fish_source.txt")
    Y = np.loadtxt("tests/testdata/pycpd/fish_target.txt")

    reg = StructurePreservedRegistration(**{"X": X, "Y": Y})
    if vis:
        fig = plt.figure()
        fig.add_axes([0, 0, 1, 1])
        callback = partial(visualize, ax=fig.axes[0])
        reg.register(callback)
        # plt.show()
    else:
        reg.register()
    T_ref = np.loadtxt("tests/testdata/pycpd/fish_deformable_2D_result_Targets.txt")
    return reg.T


def testSPR():
    T_test = runSPR()
    T_ref = np.loadtxt("tests/testdata/pycpd/fish_deformable_2D_result_Targets.txt")
    diffMat = np.linalg.norm(T_test - T_ref)
    diffSum = 1 / T_test.size * diffMat.sum()
    assert diffSum == approx(0, abs=1e-2)


if __name__ == "__main__":
    testSPR()
