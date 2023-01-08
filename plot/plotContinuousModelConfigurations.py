# plots different configruations using the wakamatsu model

import os
import sys
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib

try:
    sys.path.append(os.getcwd().replace("/tests", ""))
    from src.modelling.wakamatsuModel import (
        WakamatsuModel,
    )
    from src.visualization.plot3D import plotPointSet, plotPointSetAsLine
except:
    print("Imports for plotting continuous model failed.")
    raise


def plotConfig(
    ax,
    x0,
    aPhi,
    aTheta,
    aPsi,
    numEvalPoints,
    color,
    alpha,
):
    continuousModel = WakamatsuModel(
        **{
            "L": 1,
            "aPhi": aPhi,
            "aTheta": aTheta,
            "aPsi": aPsi,
            "x0": x0,
        }
    )
    s = np.linspace(0, 1, numEvalPoints)

    # plotPointSet(
    #     X=continuousModel.evalPositions(s),
    #     ax=ax,
    #     color=color,
    #     alpha=alpha,
    #     axisLimX=[0, 1],
    #     axisLimY=[0, 1],
    #     axisLimZ=[0, 1],
    #     waitTime=None,
    # )
    plotPointSetAsLine(
        ax=ax,
        X=continuousModel.evalPositions(s),
        color=color,
        alpha=alpha,
        axisLimX=[0, 1],
        axisLimY=[0, 1],
        axisLimZ=[0, 1],
        waitTime=None,
    )


def plotConfigsVariation_aTheta():
    steps = 10
    upperLim0 = 0.1  # use 1
    upperLim1 = 1.5  # use 3
    upperLim2 = 1.2  # use 1
    numEvalPoints = 50
    N = 10
    x0 = np.array([0.1, 0.8, 0])
    # aPhi = 1 * np.ones(N)
    # aTheta = 0 * np.ones(N) + 1 * np.linspace(0, 1, 10)
    # aPsi = 0 * np.ones(N)
    aPhi = np.zeros(N)
    aTheta = np.zeros(N)
    aPsi = np.zeros(N)
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    for i in range(0, steps):
        aTheta[0] = i * 1 / steps
        aTheta[1] = i * upperLim1 / steps
        aTheta[3] = i * upperLim2 / steps
        plotConfig(
            ax,
            x0,
            aPhi,
            aTheta,
            aPsi,
            numEvalPoints,
            color=np.array([0, 0, 1]),
            alpha=1 - (i * 0.9 / steps),
        )
    plt.show(block=True)


def plotConfigsVariation_aPhi():
    steps = 10
    upperLim0 = 0.1  # use 1
    upperLim1 = 1.5  # use 3
    upperLim2 = 1.2  # use 1
    numEvalPoints = 50
    N = 10
    x0 = np.array([0.1, 0.8, 0])
    # aPhi = 1 * np.ones(N)
    # aTheta = 0 * np.ones(N) + 1 * np.linspace(0, 1, 10)
    # aPsi = 0 * np.ones(N)
    aPhi = np.zeros(N)
    aTheta = np.zeros(N)
    aPsi = np.zeros(N)
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    for i in range(0, steps):
        aTheta[0] = 0.1
        aTheta[1] = 2
        aTheta[3] = 1
        aPhi[0] = i * -1.5 / steps
        aPhi[1] = i * -1 / steps
        plotConfig(
            ax,
            x0,
            aPhi,
            aTheta,
            aPsi,
            numEvalPoints,
            color=np.array([0, 0, 1]),
            alpha=1 - (i * 0.9 / steps),
        )
    plt.show(block=True)


def plotParallelConfigurations():
    steps = 10
    upperLim0 = 1
    upperLim1 = 1.5
    upperLim2 = 1.2
    numEvalPoints = 50
    N = 10
    x0 = np.array([0.5, 0.5, 0])
    aPhi = np.zeros(N)
    aTheta = np.zeros(N)
    aPsi = np.zeros(N)
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    for i in range(0, steps):
        x0 = np.array([0, 1 - (i * 1 / steps), 0])
        aTheta[0] = i * upperLim0 / steps
        aTheta[1] = i * upperLim1 / steps
        aTheta[3] = i * upperLim2 / steps

        plotConfig(
            ax,
            x0,
            aPhi,
            aTheta,
            aPsi,
            numEvalPoints,
            color=np.array([0, 0, 1]),
            alpha=1 - (i * 0.9 / steps),
        )
    plt.show(block=True)
    # matplotlib2tikz.save("mytikz.tex")


def plotCircularConfigurations():
    steps = 10
    upperLim0 = 0.7
    upperLim1 = -0.5  # use 3
    upperLim2 = -1  # use 1
    numEvalPoints = 50
    N = 10
    x0 = np.array([0.5, 0.5, 0])
    # aPhi = 1 * np.ones(N)
    # aTheta = 0 * np.ones(N) + 1 * np.linspace(0, 1, 10)
    # aPsi = 0 * np.ones(N)
    aPhi = np.zeros(N)
    aTheta = np.zeros(N)
    aPsi = np.zeros(N)
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    for i in range(0, steps):
        aTheta[0] = -0.2 + upperLim0
        aTheta[1] = -0.1 + i * upperLim1 / steps
        aTheta[3] = i * upperLim2 / steps
        aPhi[0] = i * -5 / steps
        aPhi[1] = i * -1 / steps
        aPhi[2] = i * -1 / steps
        # aPhi[3] = i * -1 / steps
        plotConfig(
            ax,
            x0,
            aPhi,
            aTheta,
            aPsi,
            numEvalPoints,
            color=np.array([0, 0, 1]),
            alpha=0.1 + (i * 0.1 / steps),
        )
    plt.show(block=True)


if __name__ == "__main__":
    # plotConfigsVariation_aTheta()
    # plotConfigsVariation_aPhi()

    plotParallelConfigurations()
    plotCircularConfigurations()
