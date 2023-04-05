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
    print("Imports for Neighborhood MST failed.")
    raise

# script control parameters
saveFig = False
vis = True  # if SOM iterations should be visualized
numSamples = 500
noise = 0.1
numBackgroundSamples = 100
leftBorder = -4
rightBorder = 4
lowerBorder = -2.5
upperBorder = 2.5
# L1
h = 4.3
hMin = 1.5
mu = 0.35
hAnnealing = 0.91
numSeedPoints = 70
numIterations = 100

uniSLightBlue = [0 / 255, 190 / 255, 255 / 255]


def generateSamplePoints():
    (S, s) = make_s_curve(n_samples=numSamples, noise=noise, random_state=None)
    backgroundPoints = np.column_stack(
        (
            np.random.uniform(leftBorder, rightBorder, numBackgroundSamples),
            np.random.uniform(lowerBorder, upperBorder, numBackgroundSamples),
        )
    )
    noisyS = np.vstack((S[:, [0, 2]], backgroundPoints))
    return noisyS


def computePCA(X):
    pca = PCA(n_components=1)
    pca.fit(X)

    mean = pca.mean_
    principalComponent = pca.components_
    return pca, mean, principalComponent


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


def computeL1(X):
    random_indices = random.sample(range(0, len(X)), numSeedPoints)
    seedpoints = X[random_indices, :]
    # seedpoints = np.tile(X[0, :], (len(X), 1))
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
    samplePoints = generateSamplePoints()
    pca, mean, pricipalComponent = computePCA(samplePoints)

    # PCA
    # s = np.linspace(-1, 1, 10)
    s = pca.transform(samplePoints)
    pcaLine = s * np.tile(pricipalComponent, (len(s), 1)) + mean

    # L1
    reducedPoints = computeL1(samplePoints)

    # plotting
    fig = plt.figure()
    ax = fig.add_subplot()
    plotPointSet(ax=ax, X=samplePoints, color=[0, 0, 0], size=5)
    set_axes_equal(ax=ax)
    plotPointSet(ax=ax, X=reducedPoints, color=[1, 0, 0])
    plotPoint(ax=ax, x=mean, color=uniSLightBlue)
    plt.plot(pcaLine[:, 0], pcaLine[:, 1], color=uniSLightBlue)
    plt.show(block=True)
