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
    print("Imports for Neighborhood MST failed.")
    raise

# script control parameters
saveFig = False
vis = True  # if SOM iterations should be visualized
numSamples = 500
noise = 0.1
numBackgroundSamples = 0
leftBorder = -1
rightBorder = 1
lowerBorder = -2
upperBorder = 2
# SOM
numSeedPoints = 20
alpha = 0.1
numNearestNeighbors = 2
numIterations = 100
minNumNearestNeighbors = 2
method = "kernel"
sigma2 = 0.1
sigma2Annealing = 0.93
sigma2Min = 0.01
X_SOM_iter = []


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
    # if classHandle.iteration == 3:
    #     X_SOM_3 = classHandle.T
    # elif classHandle.iteration == 5:
    #     X_SOM_5 = classHandle.T
    # elif classHandle.iteration == 10:
    #     X_SOM_10 = classHandle.T
    # elif classHandle.iteration == 30:
    #     X_SOM_30 = classHandle.T
    # elif classHandle.iteration == 50:
    #     X_SOM_50 = classHandle.T
    # elif classHandle.iteration == 100:
    #     X_SOM_100 = classHandle.T
    X_SOM_iter.append(classHandle.T)
    if savePath is not None:
        fig.savefig(savePath + fileName + "_" + str(classHandle.iter) + ".png")


def computeSOM(X):
    # random_indices = random.sample(range(0, len(X)), numSeedPoints)
    # seedpoints = X[random_indices, :]
    random_indices = random.sample(
        range(round(0.45 * len(X)), round(0.55 * len(X))), numSeedPoints
    )
    seedpoints = X[random_indices, :]
    # seedpoints = np.tile(X[0, :], (len(X), 1))
    # seedpoints = np.zeros((numSeedPoints, 2))
    seedpoints = (np.random.rand(numSeedPoints, 2) - 0.5) * 0
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
    if vis:
        visCallback = setupVisualizationCallback(testReduction)
        testReduction.registerCallback(visCallback)
    reducedPoints = testReduction.calculateReducedRepresentation()
    return reducedPoints


if __name__ == "__main__":
    samplePoints = generateSamplePoints()
    pca, mean, pricipalComponent = computePCA(samplePoints)

    # PCA
    # s = np.linspace(-1, 1, 10)
    s = pca.transform(samplePoints)
    pcaLine = s * np.tile(pricipalComponent, (len(s), 1)) + mean

    # SOM
    X_SOM = computeSOM(samplePoints)

    # plotting
    fig = plt.figure()
    ax = fig.add_subplot()
    plotPointSet(ax=ax, X=samplePoints, color=[0, 0, 0], size=5)
    set_axes_equal(ax=ax)
    plotPointSet(ax=ax, X=X_SOM, color=[1, 0, 0])
    plt.show(block=True)