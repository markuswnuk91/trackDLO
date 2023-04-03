import os
import sys
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import random
from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import shortest_path

try:
    sys.path.append(os.getcwd().replace("/tests", ""))

    from src.visualization.curveShapes3D import helixShape
    from src.visualization.plot3D import *
    from src.sensing.loadPointCloud import readPointCloudFromPLY
    from src.visualization.curveShapes3D import helixShape
    from src.utils.utils import minimalSpanningTree
except:
    print("Imports for Neighborhood MST failed.")
    raise

# script control parameters
saveFig = True
s = np.linspace(0, 1, 30)  # discretization of centerline
nSamples = 10  # num samples per discretitzed point on centerline
cov = 0.01 * np.eye(3)  # noise
distantPointIndices = (5, -5)
savePath = "/mnt/c/Users/ac129490/Documents/Dissertation/Thesis/62bebc3388a16f7dcc7f9153/figures/plots/"
saveNames = [
    "cartesianDistanceMetric.pgf",
    "shortestPathDistanceMetric.pgf",
]  # requrires as many entries as discretizations
# plot layout parameters

uniSLightBlue = [0 / 255, 190 / 255, 255 / 255]
textwidth_in_pt = 483.6969
figureScaling = 0.5
latexFontSize_in_pt = 14
latexFootNoteFontSize_in_pt = 10
desiredFigureWidth = figureScaling * textwidth_in_pt
desiredFigureHeight = figureScaling * textwidth_in_pt
tex_fonts = {
    # Use pfg for rendering
    "pgf.texsystem": "pdflatex",
    # Use LaTeX to write all text
    "text.usetex": True,
    "font.family": "serif",
    "pgf.rcfonts": False,
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": latexFontSize_in_pt,
    "font.size": latexFontSize_in_pt,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": latexFootNoteFontSize_in_pt,
    "xtick.labelsize": latexFootNoteFontSize_in_pt,
    "ytick.labelsize": latexFootNoteFontSize_in_pt,
}

if saveFig:
    matplotlib.use("pgf")
    matplotlib.rcParams.update(tex_fonts)


if __name__ == "__main__":
    # helix centerline
    helixCurve = lambda s: helixShape(s, heightScaling=1.2, frequency=1.8)

    helixCenterLine = helixCurve(s)

    # randomize points
    randomSamples = []
    for p in helixCenterLine:
        PSample = np.random.multivariate_normal(p, cov, nSamples)
        for pSample in PSample:
            randomSamples.append(pSample)
    P = np.array(randomSamples)

    minSpanTreeAdjMatrix = minimalSpanningTree(distance_matrix(P, P))
    distantPoints = P[distantPointIndices, :]
    distantPoint1 = P[distantPointIndices[0], :]
    distantPoint2 = P[distantPointIndices[1], :]
    pathDistanceMatrix, predecessorMatrix = shortest_path(
        minSpanTreeAdjMatrix,
        method="auto",
        directed=False,
        return_predecessors=True,
        unweighted=False,
        overwrite=False,
        indices=distantPointIndices[0],
    )
    # # plot point set
    # fig = plt.figure()
    # ax = fig.add_subplot(projection="3d")
    # for p in P:
    #     alpha = (p[0] - np.min(P[:, 0])) / (np.max(P[:, 0]) - np.min(P[:, 0]))
    #     alpha = 0.3 * alpha + 0.1
    #     plotPoint(x=p, ax=ax, color=[0, 0, 0], size=5, alpha=alpha)
    # set_axes_equal(ax)

    # # plot minSpanTree
    # fig = plt.figure()
    # ax = fig.add_subplot(projection="3d")
    # pointPair_indices = np.nonzero(minSpanTreeAdjMatrix)
    # for i in range(0, len(pointPair_indices[0])):
    #     pointPair = P[(pointPair_indices[0][i], pointPair_indices[1][i]), :]
    #     alpha = (((pointPair[0][0] + pointPair[1][0]) / 2) - np.min(P[:, 0])) / (
    #         np.max(P[:, 0]) - np.min(P[:, 0])
    #     )
    #     alpha = 0.3 * alpha + 0.1
    #     plotLine(ax, pointPair, color=[0, 0, 0], alpha=alpha)

    # plot cartesian distance metric
    fig, ax = setupLatexPlot3D(
        figureWidth=desiredFigureWidth,
        figureHeight=desiredFigureHeight,
        axisLimX=[-1.2, 1.2],
        axisLimY=[-1.2, 1.2],
        axisLimZ=[-0.6, 1.8],
        viewAngle=(20, 30),
        xTickStep=0.5,
        yTickStep=0.5,
        zTickStep=0.5,
    )
    alphaGain = 0.0
    alphaOffset = 0.5
    for p in P:
        alpha = (p[0] - np.min(P[:, 0])) / (np.max(P[:, 0]) - np.min(P[:, 0]))
        alpha = alphaGain * alpha + alphaOffset
        plotPoint(x=p, ax=ax, color=[0, 0, 0], size=5, alpha=alpha, edgeColor="none")
    plotPoint(ax=ax, x=distantPoint1, color=uniSLightBlue, size=20)
    plotPoint(ax=ax, x=distantPoint2, color=uniSLightBlue, size=20)
    plotLine(ax, distantPoints, color=uniSLightBlue, linewidth=3)
    # create legend
    # cartesianError = mlines.Line2D(
    #     [], [], color=uniSLightBlue, label="cartesian distance"
    # )
    # plt.legend(
    #     handles=[cartesianError],
    #     loc="upper right",
    #     bbox_to_anchor=(1, 0.8),
    # )
    ax.set_xticks(np.arange(-1.2, 1.2, step=0.5))
    ax.set_yticks(np.arange(-1.2, 1.2, step=0.5))
    ax.set_zticks(np.arange(0, 2.01, step=0.5))
    if saveFig:
        plt.savefig(
            savePath + saveNames[0],
            bbox_inches="tight",
        )

    # plot path length metric
    fig, ax = setupLatexPlot3D(
        figureWidth=desiredFigureWidth,
        figureHeight=desiredFigureHeight,
        axisLimX=[-1.2, 1.2],
        axisLimY=[-1.2, 1.2],
        axisLimZ=[-0.4, 2],
        viewAngle=(20, 30),
        xTickStep=0.5,
        yTickStep=0.5,
        zTickStep=0.5,
    )
    alphaGain = 0.0
    alphaOffset = 0.5
    pointPair_indices = np.nonzero(minSpanTreeAdjMatrix)
    for i in range(0, len(pointPair_indices[0])):
        pointPair = P[(pointPair_indices[0][i], pointPair_indices[1][i]), :]
        alpha = (((pointPair[0][0] + pointPair[1][0]) / 2) - np.min(P[:, 0])) / (
            np.max(P[:, 0]) - np.min(P[:, 0])
        )
        alpha = alphaGain * alpha + alphaOffset
        plotLine(ax, pointPair, color=[0, 0, 0], alpha=alpha)
    plotPoint(ax=ax, x=distantPoint1, color=[1, 0, 0], size=20)
    plotPoint(ax=ax, x=distantPoint2, color=[1, 0, 0], size=20)
    currentIdx = distantPointIndices[1]
    predecessorIdx = predecessorMatrix[currentIdx]
    while predecessorIdx != distantPointIndices[0]:
        pointPair = P[(currentIdx, predecessorIdx), :]
        alpha = (((pointPair[0][0] + pointPair[1][0]) / 2) - np.min(P[:, 0])) / (
            np.max(P[:, 0]) - np.min(P[:, 0])
        )
        alpha = 0.8 * alpha + 0.2
        plotLine(ax, pointPair, color=[1, 0, 0], linewidth=3, alpha=alpha)
        currentIdx = predecessorIdx
        predecessorIdx = predecessorMatrix[currentIdx]

    # create legend
    # shortestPathError = mlines.Line2D(
    #     [], [], color=[1, 0, 0], label="shortest path distance"
    # )
    # plt.legend(
    #     handles=[shortestPathError],
    #     loc="upper right",
    #     bbox_to_anchor=(1, 0.8),
    # )
    ax.set_xticks(np.arange(-1.2, 1.2, step=0.5))
    ax.set_yticks(np.arange(-1.2, 1.2, step=0.5))
    ax.set_zticks(np.arange(0, 2.01, step=0.5))

    if saveFig:
        plt.savefig(
            savePath + saveNames[1],
            bbox_inches="tight",
        )
    else:
        plt.show(block=True)
