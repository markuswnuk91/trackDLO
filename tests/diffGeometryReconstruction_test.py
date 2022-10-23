import os
import sys
import argparse
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
from pytest import approx

try:
    sys.path.append(os.getcwd().replace("/tests", ""))
    from src.reconstruction.differentialGeometry.differentialGeometryReconstruction import (
        DifferentialGeometryReconstruction,
    )
except:
    print("Imports for DifferentialGeometryReconstruction failed.")
    raise
vis = True  # enable for visualization


def visualize(X, Y, ax):
    plt.cla()
    ax.scatter(X[:, 0], X[:, 1], color="red", label="Target")
    ax.scatter(Y[:, 0], Y[:, 1], color="blue", label="Source")
    plt.text(
        0.7,
        0.92,
        "Wakamatsu Model Reconstruction",
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax.transAxes,
        fontsize="x-large",
    )
    ax.legend(loc="upper left", fontsize="x-large")
    plt.draw()
    plt.pause(0.001)


def runReconstruction():
    X = np.ones((10, 3))
    X[:, 2] = np.linspace(0, 100, 10)
    X[5, 2] = 55
    Sx = np.linspace(0, 1, 10)
    testReconstruction = DifferentialGeometryReconstruction(
        **{"X": X, "Sx": Sx, "L": 100}
    )
    testReconstruction.estimateShape()


if __name__ == "__main__":
    runReconstruction()
