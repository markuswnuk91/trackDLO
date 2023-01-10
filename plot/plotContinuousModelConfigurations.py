# plots different configruations using the wakamatsu model

import os
import sys
from functools import partial
import matplotlib
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

# plot control
saveFigs = True
savePath = "/mnt/c/Users/ac129490/Documents/Dissertation/Thesis/62bebc3388a16f7dcc7f9153/figures/"

textwidth_in_pt = 483.6969
figureScaling = 1
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
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
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


def set_size(width, height=None, fraction=1, subplots=(1, 1)):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == "thesis":
        width_pt = 426.79135
    elif width == "beamer":
        width_pt = 307.28987
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    if height is None:
        fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])
    else:
        fig_height_in = height * inches_per_pt
    return (fig_width_in, fig_height_in)


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

    fig = plt.figure(
        figsize=set_size(width=desiredFigureWidth, height=desiredFigureHeight)
    )
    ax = fig.add_subplot(projection="3d")
    labels = []
    for i in range(0, steps):
        x0 = np.array([0, 1 - (i * 1 / (steps + 1)), 0])
        aTheta[0] = i * upperLim0 / steps
        aTheta[1] = i * upperLim1 / steps
        aTheta[3] = i * upperLim2 / steps
        labels.append("$\kappa_{{max}} = {}$".format(str(i)))
        plotConfig(
            ax,
            x0,
            aPhi,
            aTheta,
            aPsi,
            numEvalPoints,
            color=np.array([0, 0, 0]),
            alpha=1 - (i * 0.7 / steps),
        )
    # adjust viewing settings
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    plt.legend(labels, loc="right", bbox_to_anchor=(1.1, 0.55))
    ax.view_init(15, -115)
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    if saveFigs:
        plt.savefig(savePath + "matplotLibTest.pgf", bbox_inches="tight")
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
    fig = plt.figure(figsize=set_size(textwidth_in_pt))
    ax = fig.add_subplot(projection="3d")
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
            color=np.array([0, 0, 1]),
            alpha=1 - (i * 0.7 / steps),
        )
    # adjust viewing settings
    ax.view_init(15, -115)
    plt.show(block=True)


if __name__ == "__main__":
    # plotConfigsVariation_aTheta()
    # plotConfigsVariation_aPhi()

    plotIncreasingCurvatureConfigurations()
    plotIncresingTorsionConfigurations()
