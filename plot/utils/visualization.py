import os
import sys
import argparse
import matplotlib.pyplot as plt
from functools import partial
import numpy as np


def visualizePointSets(
    X,
    Y,
    ax,
    axisLimX=[0, 1],
    axisLimY=[0, 1],
    axisLimZ=[0, 1],
    savePath=None,
    fileName="img.png",
):
    if savePath is not None and type(savePath) is not str:
        raise ValueError("Error saving 3D plot. The given path should be a string.")

    if fileName is not None and type(fileName) is not str:
        raise ValueError("Error saving 3D plot. The given filename should be a string.")

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
    ax.set_xlim(axisLimX[0], axisLimX[1])
    ax.set_ylim(axisLimY[0], axisLimY[1])
    ax.set_zlim(axisLimZ[0], axisLimZ[1])
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$z$")
    plt.draw()
    plt.pause(0.001)
    if savePath is not None:
        plt.savefig(savePath + fileName)


def setupVisualizationCallback(
    axisLimX=[0, 1], axisLimY=[0, 1], axisLimZ=[0, 1], savePath=None, fileName="img.png"
):
    if savePath is not None and type(savePath) is not str:
        raise ValueError("Error saving 3D plot. The given path should be a string.")

    if fileName is not None and type(fileName) is not str:
        raise ValueError("Error saving 3D plot. The given filename should be a string.")

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    return partial(
        visualizePointSets,
        ax=ax,
        axisLimX=axisLimX,
        axisLimY=axisLimY,
        axisLimZ=axisLimZ,
        savePath=savePath,
    )
