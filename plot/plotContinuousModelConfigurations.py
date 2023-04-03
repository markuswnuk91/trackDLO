# plots different configruations using the wakamatsu model
# figure in chapter 4 modelling, continuous shape representation

import os
import sys
from functools import partial
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
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
    print("Imports for plotting continuous model co0nfigurations failed.")
    raise

# plot control
saveFigs = False
savePath = "/mnt/c/Users/ac129490/Documents/Dissertation/Thesis/62bebc3388a16f7dcc7f9153/figures/"

colorMap = matplotlib.colormaps["viridis"]
textwidth_in_pt = 483.6969
figureScaling = 0.45
latexFontSize_in_pt = 14
latexFootNoteFontSize_in_pt = 10
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
    "legend.fontsize": latexFootNoteFontSize_in_pt,
    "xtick.labelsize": latexFootNoteFontSize_in_pt,
    "ytick.labelsize": latexFootNoteFontSize_in_pt,
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
    model,
    kappaUpperLimit,
    numEvalPoints,
    alpha,
):
    s = np.linspace(0, 1, numEvalPoints)
    kappa = np.sqrt(model.evalKappaSquared(s))
    # normalize kappa to range from 255 to 1
    kappaColor = kappa / kappaUpperLimit
    plotPointSetAsColorGradedLine(
        ax=ax,
        X=model.evalPositions(s),
        colorMap=colorMap,
        colorGrad=kappaColor,
        alpha=alpha,
        linewidth=3.0,
        waitTime=None,
    )


def plotConfigWithColorGradedTorsion(
    ax,
    model,
    omegaUpperLimit,
    numEvalPoints,
    alpha,
):
    s = np.linspace(0, 1, numEvalPoints)
    omega = np.sqrt(model.evalOmegaSquared(s))
    # normalize kappa to range from 255 to 1
    omegaColor = omega / omegaUpperLimit
    plotPointSetAsColorGradedLine(
        ax=ax,
        X=model.evalPositions(s),
        colorMap=colorMap,
        colorGrad=omegaColor,
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

    fig, ax = setupLatexPlot3D(
        figureWidth=desiredFigureWidth, figureHeight=desiredFigureHeight
    )
    labels = []
    models = []
    kappaUpperLimit = 0
    for i in range(0, steps):
        x0 = np.array([0, 1 - (i * 1 / (steps + 1)), 0])
        aTheta[0] = i * upperLim0 / steps
        aTheta[1] = i * upperLim1 / steps
        aTheta[3] = i * upperLim2 / steps
        models.append(
            WakamatsuModel(
                **{
                    "L": 1,
                    "aPhi": aPhi.copy(),
                    "aTheta": aTheta.copy(),
                    "aPsi": aPsi.copy(),
                    "x0": x0.copy(),
                }
            )
        )
    for model in models:
        kappaMax = np.max(np.sqrt(model.evalKappaSquared(np.linspace(0, 1, 1000))))
        if kappaMax > kappaUpperLimit:
            kappaUpperLimit = kappaMax
    for model in models:
        plotConfigWithColorGradedCurvature(
            ax=ax,
            model=model,
            numEvalPoints=100,
            kappaUpperLimit=kappaUpperLimit,
            alpha=1 - (i * 0.7 / steps),
        )

    # plt.legend(labels, loc="right", bbox_to_anchor=(1.05, 0.55))

    # colormap
    lowerLim = 0
    upperLim = np.round(kappaUpperLimit)
    norm = matplotlib.colors.Normalize(vmin=lowerLim, vmax=upperLim)  # Normalizer
    sm = plt.cm.ScalarMappable(cmap=colorMap, norm=norm)  # creating ScalarMappable
    # sm.set_array([])
    cbar = plt.colorbar(
        sm,
        ticks=np.linspace(0, upperLim, 2),
        location="right",
        anchor=(-0.3, 0.5),
        shrink=0.5,
    )
    cbar.ax.set_ylabel(
        "curvature",
        rotation=270,
        labelpad=0,
        fontsize=latexFootNoteFontSize_in_pt,
    )
    cbar.ax.set_yticklabels(["low", "high"], fontsize=latexFootNoteFontSize_in_pt)
    # cbar.ax.yaxis.set_major_formatter(tick.FormatStrFormatter("%.1f"))

    # ax.set_xticks(np.arange(0, 1.1, step=0.5))
    # ax.set_yticks(np.arange(0, 1.1, step=0.5))
    # ax.set_zticks(np.arange(0, 1.1, step=0.5))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    # reduce label padding
    ax.xaxis.labelpad = -10
    ax.yaxis.labelpad = -10
    ax.zaxis.labelpad = -10

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
    fig, ax = setupLatexPlot3D(
        figureWidth=desiredFigureWidth, figureHeight=desiredFigureHeight
    )
    labels = []
    models = []
    omegaUpperLimit = 0
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
        models.append(
            WakamatsuModel(
                **{
                    "L": 1,
                    "aPhi": aPhi.copy(),
                    "aTheta": aTheta.copy(),
                    "aPsi": aPsi.copy(),
                    "x0": x0.copy(),
                }
            )
        )

    for model in models:
        omegaMax = np.max(np.sqrt(model.evalOmegaSquared(np.linspace(0, 1, 1000))))
        if omegaMax > omegaUpperLimit:
            omegaUpperLimit = omegaMax
    for model in models:
        plotConfigWithColorGradedTorsion(
            ax=ax,
            model=model,
            numEvalPoints=100,
            omegaUpperLimit=omegaUpperLimit,
            alpha=1 - (i * 0.7 / steps),
        )

    # colormap
    lowerLim = 0
    upperLim = np.round(omegaUpperLimit)
    norm = matplotlib.colors.Normalize(vmin=lowerLim, vmax=upperLim)  # Normalizer
    sm = plt.cm.ScalarMappable(cmap=colorMap, norm=norm)  # creating ScalarMappable
    # sm.set_array([])
    cbar = plt.colorbar(
        sm,
        ticks=np.linspace(0, upperLim, 2),
        location="right",
        anchor=(-0.3, 0.5),
        shrink=0.5,
    )
    cbar.ax.yaxis.set_major_formatter(tick.FormatStrFormatter("%.1f"))
    cbar.ax.set_ylabel(
        "torsion",
        rotation=270,
        labelpad=0,
        fontsize=latexFootNoteFontSize_in_pt,
    )
    cbar.ax.set_yticklabels(["low", "high"], fontsize=latexFootNoteFontSize_in_pt)

    # ax.set_xticks(np.arange(0, 1.1, step=0.5))
    # ax.set_yticks(np.arange(0, 1.1, step=0.5))
    # ax.set_zticks(np.arange(0, 1.1, step=0.5))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    # reduce label padding
    ax.xaxis.labelpad = -10
    ax.yaxis.labelpad = -10
    ax.zaxis.labelpad = -10

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
