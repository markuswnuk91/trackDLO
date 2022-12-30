import os
import sys
import argparse
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
from pytest import approx

try:
    sys.path.append(os.getcwd().replace("/tests", ""))
    from src.reconstruction.continuous.continuousReconstruction import (
        ContinuousReconstruction,
    )
except:
    print("Imports for ContinuousReconstruction failed.")
    raise
vis = True  # enable for visualization


def visualize(X, Y, ax):
    plt.cla()
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], color="blue", label="Source")
    ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], color="red", label="Target")
    # plt.text(
    #     0.7,
    #     0.92,
    #     s="Wakamatsu Model Reconstruction",
    #     horizontalalignment="center",
    #     verticalalignment="center",
    #     transform=ax.transAxes,
    #     fontsize="x-large",
    # )
    ax.legend(loc="upper left", fontsize="x-large")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_zlim(0, 100)
    plt.draw()
    plt.pause(0.001)


def runReconstruction():
    Y = np.ones((10, 3))
    Y[:, 1] = np.linspace(0, 100, 10)
    # Y[:, 2] = np.linspace(0, 50, 10)
    # X[5, 2] = 55
    SY = np.linspace(0, 100, 10) / 100
    if vis:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        callback = partial(visualize, ax=ax)
        testReconstruction = ContinuousReconstruction(
            **{
                "Y": Y,
                "SY": SY,
                "L": 100,
                "numSc": 100,
                "Rflex": 1,
                "Rtor": 1,
                "Roh": 0,
                "callback": callback,
            }
        )
    else:
        testReconstruction = ContinuousReconstruction(
            **{"Y": Y, "SY": SY, "L": 100, "numSc": 100}
        )
    testReconstruction.reconstructShape()

    testReconstruction


if __name__ == "__main__":
    runReconstruction()
