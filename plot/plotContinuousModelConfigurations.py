# plots different configruations using the wakamatsu model

import os
import sys
from functools import partial
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

try:
    sys.path.append(os.getcwd().replace("/tests", ""))
    from src.modelling.wakamatsuModel import (
        WakamatsuModel,
    )
    from src.visualization.plot3D import (
        setupLatexPlot3D,
        plotPointSet,
        plotPointSetAsLine,
        plotPointSetAsColorGradedLine,
    )
except:
    print("Imports for plotting continuous model failed.")
    raise

# plot control
saveFigs = False
savePath = "/mnt/c/Users/ac129490/Documents/Dissertation/Thesis/62bebc3388a16f7dcc7f9153/figures/"

colorMap = matplotlib.colormaps["viridis"]
textwidth_in_pt = 483.6969
figureScaling = 0.45
latexFontSize_in_pt = 14

desiredFigureWidth = figureScaling * textwidth_in_pt
desiredFigureHeight = figureScaling * textwidth_in_pt
tex_fonts = {
    #    "pgf.texsystem": "pdflatex",
    # Use LaTeX to write all text
    "text.usetex": True,
    "font.family": "serif",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": latexFontSize_in_pt,
    "font.size": latexFontSize_in_pt,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 8,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
}

if saveFigs:
    matplotlib.use("pgf")
    matplotlib.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "font.family": "serif",
            "text.usetex": True,
            "pgf.rcfonts": False,
        }
    )
    matplotlib.rcParams.update(tex_fonts)


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
        linewidth=3.0,
        waitTime=None,
    )


def plotConfigWithColorGradedCurvature(
    ax,
    x0,
    aPhi,
    aTheta,
    aPsi,
    numEvalPoints,
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
    kappa = np.sqrt(continuousModel.evalKappaSquared(s))
    # normalize kappa to range from 255 to 1
    kappaColor = kappa / np.max(kappa)
    plotPointSetAsColorGradedLine(
        ax=ax,
        X=continuousModel.evalPositions(s),
        colorMap=colorMap,
        colorGrad=kappaColor,
        alpha=alpha,
        linewidth=3.0,
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


def plotIncreasingCurvatureConfigurations():
    steps = 9
    upperLim0 = 1
    upperLim1 = 1.5
    upperLim2 = 1.2
    numEvalPoints = 50
    N = 10
    x0 = np.array([0.5, 0.5, 0])
    aPhi = np.zeros(N)
    aTheta = np.zeros(N)
    aPsi = np.zeros(N)

    fig, ax = setupLatexPlot3D()
    labels = []
    for i in range(0, steps):
        x0 = np.array([0, 1 - (i * 1 / (steps + 1)), 0])
        aTheta[0] = i * upperLim0 / steps
        aTheta[1] = i * upperLim1 / steps
        aTheta[3] = i * upperLim2 / steps
        labels.append("$\kappa_{{max}} = {}$".format(str(i)))
        plotConfigWithColorGradedCurvature(
            ax,
            x0,
            aPhi,
            aTheta,
            aPsi,
            numEvalPoints,
            alpha=1 - (i * 0.7 / steps),
        )

    plt.legend(labels, loc="right", bbox_to_anchor=(1.05, 0.55))
    ax.view_init(15, -115)
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    if saveFigs:
        plt.savefig(
            savePath + "ContinuousModelConfigurations_increasingCurvature.pgf",
            bbox_inches="tight",
        )
    plt.show(block=True)


def plotIncresingTorsionConfigurations():
    steps = 9
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
    fix, ax = setupLatexPlot3D()
    labels = []
    for i in range(0, steps):
        x0 = np.array([0, 1 - (i * 1 / (steps + 1)), 0])
        # aTheta[0] = -0.2 + upperLim0
        # aTheta[1] = -0.1 + i * upperLim1 / steps
        # aTheta[3] = i * upperLim2 / steps
        # aPhi[0] = i * -5 / steps
        # aPhi[1] = i * -1 / steps
        # aPhi[2] = i * -1 / steps
        # aPhi[3] = i * -1 / steps
        aTheta[0] = i * 1 / steps
        aTheta[1] = i * 1.5 / steps
        aTheta[3] = i * 1.2 / steps
        aPhi[0] = i * 0 / steps
        aPhi[1] = i * -1.5 / steps
        aPhi[2] = i * -1.5 / steps
        aPhi[3] = i * 0.1 / steps
        plotConfig(
            ax,
            x0,
            aPhi,
            aTheta,
            aPsi,
            numEvalPoints,
            color=matplotlib.colormaps["viridis"](i / steps)[:3],
            alpha=1 - (i * 0.7 / steps),
        )
        labels.append("$\omega_{{max}}= {}$".format(str(i), str(i)))
    plt.legend(labels, loc="right", bbox_to_anchor=(1.05, 0.55))
    ax.view_init(15, -115)
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    if saveFigs:
        plt.savefig(
            savePath + "ContinuousModelConfigurations_increasingTorsion.pgf",
            bbox_inches="tight",
        )
    plt.show(block=True)
    plt.show(block=True)


if __name__ == "__main__":
    # plotConfigsVariation_aTheta()
    # plotConfigsVariation_aPhi()

    plotIncreasingCurvatureConfigurations()
    plotIncresingTorsionConfigurations()
