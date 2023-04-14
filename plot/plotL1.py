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
    from src.localization.downsampling.l1median.l1Median import L1Median
    from src.visualization.curveShapes3D import helixShape
    from src.visualization.plot3D import *
    from src.sensing.loadPointCloud import readPointCloudFromPLY
    from src.visualization.curveShapes3D import helixShape
except:
    print("Imports for plotting L1 example failed.")
    raise

# script control parameters
saveFig = True
vis = False  # if iterations should be visualized
savePath = "/mnt/c/Users/ac129490/Documents/Dissertation/Thesis/62bebc3388a16f7dcc7f9153/figures/plots/"
fileName = "plotExampleL1"
fileType = ".pdf"
numSamples = 500
numOutliers = 100
noise = 0.1
leftBorder = -1.3
rightBorder = 1.3
lowerBorder = -2.3
upperBorder = 2.3
X_iter = []

# parameters
h = 5
hMin = 1.4
muList = [0, 0.1, 0.35]
hAnnealing = 0.95
numSeedPoints = 50
numIterations = 150

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
    if savePath is not None:
        fig.savefig(savePath + fileName + "_" + str(classHandle.iter) + ".png")


def computeL1(X, seedpoints, mu):
    testReduction = L1Median(
        **{
            "Y": X,
            "X": seedpoints,
            "h": h,
            "hMin": hMin,
            "mu": mu,
            "hAnnealing": hAnnealing,
            "max_iterations": numIterations,
        }
    )
    if vis:
        visCallback = setupVisualizationCallback(testReduction)
        testReduction.registerCallback(visCallback)
    return testReduction.calculateReducedRepresentation()


if __name__ == "__main__":
    samplePoints = generateSamplePoints(0)
    samplePointsOutliers = generateSamplePoints(numOutliers)
    random_indices = random.sample(range(0, len(samplePoints)), numSeedPoints)
    seedpoints = samplePoints[random_indices, :]
    # seedpoints = np.tile(X[0, :], (len(X), 1))

    # L1 for different mu
    for mu in muList:
        X_L1 = computeL1(samplePoints, seedpoints.copy(), mu)
        X_iter.append(X_L1)
    # L1 with outliers
    X_Outlier = computeL1(samplePointsOutliers, seedpoints.copy(), muList[-1])

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
    # results for diffrent mu
    for i, X_L1 in enumerate(X_iter):
        fig, ax = setupPlotLayout()
        plotPointSet(ax=ax, X=samplePoints, color=[0, 0, 0], size=samplePointSize)
        set_axes_equal(ax=ax)
        plotPointSet(ax=ax, X=X_L1, color=[1, 0, 0], size=seedPointSize)
        plt.grid(False)
        plt.axis("off")
        plt.show(block=False)
        if saveFig:
            plt.savefig(
                savePath + fileName + "_" + str(muList[i]) + fileType,
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
