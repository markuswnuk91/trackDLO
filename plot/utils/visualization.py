import os
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np


def visualizePointSets(
    X,
    Y,
    ax,
    axisLimX=[0, 1],
    axisLimY=[0, 1],
    axisLimZ=[0, 1],
    waitTime=0.001,
):
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
    ax.set_xlim(axisLimX[0], axisLimX[1])
    ax.set_ylim(axisLimY[0], axisLimY[1])
    ax.set_zlim(axisLimZ[0], axisLimZ[1])
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$z$")
    if waitTime is not None or -1:
        plt.draw()
        plt.pause(waitTime)
    elif waitTime is None:
        plt.show(blocking=None)
    elif waitTime is -1:
        plt.show(blocking=True)