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
s = np.linspace(0, 1, 30)  # discretization of centerline
nSamples = 10  # num samples per discretitzed point on centerline
cov = 0.01 * np.eye(3)  # noise
distantPointIndices = (5, -5)
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

    # plot point set
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    for p in P:
        alpha = (p[0] - np.min(P[:, 0])) / (np.max(P[:, 0]) - np.min(P[:, 0]))
        alpha = 0.3 * alpha + 0.1
        plotPoint(x=p, ax=ax, color=[0, 0, 0], size=5, alpha=alpha)
    set_axes_equal(ax)

    # plot minSpanTree
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    pointPair_indices = np.nonzero(minSpanTreeAdjMatrix)
    for i in range(0, len(pointPair_indices[0])):
        pointPair = P[(pointPair_indices[0][i], pointPair_indices[1][i]), :]
        alpha = (((pointPair[0][0] + pointPair[1][0]) / 2) - np.min(P[:, 0])) / (
            np.max(P[:, 0]) - np.min(P[:, 0])
        )
        alpha = 0.3 * alpha + 0.1
        plotLine(ax, pointPair, color=[0, 0, 0], alpha=alpha)

    # plot cartesian distance vs path length
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

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    plotPointSet(X=P, ax=ax, color=[0, 0, 0], size=5)
    pointPair_indices = np.nonzero(minSpanTreeAdjMatrix)
    for i in range(0, len(pointPair_indices[0])):
        pointPair = P[(pointPair_indices[0][i], pointPair_indices[1][i]), :]
        alpha = (((pointPair[0][0] + pointPair[1][0]) / 2) - np.min(P[:, 0])) / (
            np.max(P[:, 0]) - np.min(P[:, 0])
        )
        alpha = 0.3 * alpha + 0.1
        plotLine(ax, pointPair, color=[0, 0, 0], alpha=alpha)
    plotLine(ax, distantPoints, color=[1, 0, 0])
    plotPoint(ax=ax, x=distantPoint1, color=[1, 0, 0], size=20)
    plotPoint(ax=ax, x=distantPoint2, color=[1, 0, 0], size=20)
    currentIdx = distantPointIndices[1]
    predecessorIdx = predecessorMatrix[currentIdx]
    while predecessorIdx != distantPointIndices[0]:
        pointPair = P[(currentIdx, predecessorIdx), :]
        plotLine(ax, pointPair, color=[0, 0, 1])
        currentIdx = predecessorIdx
        predecessorIdx = predecessorMatrix[currentIdx]
    plt.show(block=True)
