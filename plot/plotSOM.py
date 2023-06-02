import os
import sys
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import random
from sklearn.datasets import make_s_curve
from sklearn.decomposition import PCA
from functools import partial

try:
    sys.path.append(os.getcwd().replace("/tests", ""))
    from src.localization.downsampling.som.som import SelfOrganizingMap
    from src.visualization.curveShapes3D import helixShape
    from src.visualization.plot3D import *
    from src.sensing.loadPointCloud import readPointCloudFromPLY
    from src.visualization.curveShapes3D import helixShape
except:
    print("Imports for plotting SOM Example failed.")
    raise

# script control parameters
saveFig = False
vis = False  # if iterations should be visualized
savePath = "/mnt/c/Users/ac129490/Documents/Dissertation/Thesis/62bebc3388a16f7dcc7f9153/figures/plots/"
fileName = "plotExampleSOM"
fileType = ".pdf"
numSamples = 500
numOutliers = 100
noise = 0.1
numBackgroundSamples = 0
leftBorder = -1.3
rightBorder = 1.3
lowerBorder = -2.3
upperBorder = 2.3
X_SOM_iter = []
plotIterList = [5, 10, 30]

# parameters
numSeedPoints = 20
alpha = 0.1
numNearestNeighbors = 2
numIterations = 100
minNumNearestNeighbors = 2
method = "kernel"
sigma2 = 0.5
sigma2Annealing = 0.93
sigma2Min = 0.2


# plot layout parameters
samplePointSize = 5
seedPointSize = 50
uniSLightBlue = [0 / 255, 190 / 255, 255 / 255]
textwidth_in_pt = 483.6969
figureScaling = 1
latexFontSize_in_pt = 14
latexFootNoteFontSize_in_pt = 10
desiredFigureWidth = figureScaling * textwidth_in_pt
desiredFigureHeight = None
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


def setupPlotLayout():
    fig, ax = setupLatexPlot2D(
        figureWidth=desiredFigureWidth,
        figureHeight=desiredFigureHeight,
        axisLimX=[leftBorder, rightBorder],
        axisLimY=[lowerBorder, upperBorder],
        xTickStep=0.5,
        yTickStep=0.5,
    )
    return fig, ax


def generateSamplePoints(numBackgroundSamples):
    (S, s) = make_s_curve(n_samples=numSamples, noise=noise, random_state=None)
    backgroundPoints = np.column_stack(
        (
            np.random.uniform(leftBorder, rightBorder, numBackgroundSamples),
            np.random.uniform(lowerBorder, upperBorder, numBackgroundSamples),
        )
    )
    noisyS = np.vstack((S[:, [0, 2]], backgroundPoints))
    return noisyS


def setupVisualizationCallback(classHandle):
    fig = plt.figure()
    ax = fig.add_subplot()
    return partial(
        visualizationCallback,
        fig,
        ax,
        classHandle,
        # savePath="/mnt/c/Users/ac129490/Documents/Dissertation/Software/trackdlo/imgs/continuousShapeReconstuction/helix_fail3/",
    )


def visualizationCallback(
    fig,
    ax,
    classHandle,
    savePath=None,
    fileName="img",
):
    if savePath is not None and type(savePath) is not str:
        raise ValueError("Error saving 3D plot. The given path should be a string.")

    if fileName is not None and type(fileName) is not str:
        raise ValueError("Error saving 3D plot. The given filename should be a string.")
    if vis:
        plt.cla()
        ax.set_xlim(leftBorder, rightBorder)
        ax.set_ylim(lowerBorder, upperBorder)
        plotPointSets(
            ax=ax,
            X=classHandle.T,
            Y=classHandle.Y,
            ySize=5,
            xSize=10,
            # yMarkerStyle=".",
            yAlpha=0.01,
        )
        plt.draw()
        plt.pause(0.1)
    print(classHandle.iteration)
    X_SOM_iter.append(classHandle.T.copy())
    if savePath is not None:
        fig.savefig(savePath + fileName + "_" + str(classHandle.iter) + ".png")


def computeSOM(X, seedpoints):
    testReduction = SelfOrganizingMap(
        **{
            "Y": X,
            "X": seedpoints,
            "alpha": alpha,
            "numNearestNeighbors": numNearestNeighbors,
            "max_iterations": numIterations,
            "minNumNearestNeighbors": minNumNearestNeighbors,
            "method": method,
            "sigma2": sigma2,
            "sigma2Annealing": sigma2Annealing,
            "sigma2Min": sigma2Min,
        }
    )

    visCallback = setupVisualizationCallback(testReduction)
    testReduction.registerCallback(visCallback)
    reducedPoints = testReduction.calculateReducedRepresentation()
    return reducedPoints


def computeSOM_noCallback(X, seedpoints):
    testReduction = SelfOrganizingMap(
        **{
            "Y": X,
            "X": seedpoints,
            "alpha": alpha,
            "numNearestNeighbors": numNearestNeighbors,
            "max_iterations": numIterations,
            "minNumNearestNeighbors": minNumNearestNeighbors,
            "method": method,
            "sigma2": sigma2,
            "sigma2Annealing": sigma2Annealing,
            "sigma2Min": sigma2Min,
        }
    )
    reducedPoints = testReduction.calculateReducedRepresentation()
    return reducedPoints


if __name__ == "__main__":
    samplePoints = generateSamplePoints(0)
    samplePointsOutliers = generateSamplePoints(numOutliers)

    # generate random seedpoints
    # random_indices = random.sample(range(0, len(X)), numSeedPoints)
    # seedpoints = X[random_indices, :]
    random_indices = random.sample(
        range(round(0.45 * len(samplePoints)), round(0.55 * len(samplePoints))),
        numSeedPoints,
    )
    seedpoints = samplePoints[random_indices, :]
    # seedpoints = np.tile(X[0, :], (len(X), 1))
    # seedpoints = np.zeros((numSeedPoints, 2))
    seedpoints = (np.random.rand(numSeedPoints, 2) - 0.5) * 0.01

    # SOM
    X_SOM = computeSOM(samplePoints, seedpoints.copy())
    X_Outlier = computeSOM_noCallback(samplePoints, seedpoints.copy())
    # plotting
    # seedpoints
    fig, ax = setupPlotLayout()
    plotPointSet(ax=ax, X=samplePoints, color=[0, 0, 0], size=samplePointSize)
    set_axes_equal(ax=ax)
    plotPointSet(ax=ax, X=seedpoints, color=[1, 0, 0], size=seedPointSize)
    plt.grid(False)
    plt.axis("off")
    if saveFig:
        plt.savefig(
            savePath + fileName + "_" + "init" + fileType,
            bbox_inches="tight",
        )
    else:
        plt.show(block=False)
    # iterations
    for i in plotIterList:
        fig, ax = setupPlotLayout()
        plotPointSet(ax=ax, X=samplePoints, color=[0, 0, 0], size=samplePointSize)
        set_axes_equal(ax=ax)
        plotPointSet(ax=ax, X=X_SOM_iter[i], color=[1, 0, 0], size=seedPointSize)
        plt.grid(False)
        plt.axis("off")
        plt.show(block=False)
        if saveFig:
            plt.savefig(
                savePath + fileName + "_" + str(i) + fileType,
                bbox_inches="tight",
            )
        else:
            plt.show(block=False)
    # result for outliers
    fig, ax = setupPlotLayout()
    plotPointSet(ax=ax, X=samplePointsOutliers, color=[0, 0, 0], size=samplePointSize)
    set_axes_equal(ax=ax)
    plotPointSet(ax=ax, X=X_Outlier, color=[1, 0, 0], size=seedPointSize)
    plt.grid(False)
    plt.axis("off")
    plt.show(block=False)
    if saveFig:
        plt.savefig(
            savePath + fileName + "_" + "outlier" + fileType,
            bbox_inches="tight",
        )
    else:
        plt.show(block=False)
    if not saveFig:
        plt.show(block=True)
