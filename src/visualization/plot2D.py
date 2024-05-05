import sys, os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from warnings import warn

try:
    sys.path.append(os.getcwd().replace("/src/visualization", ""))
    from src.visualization.plotUtils import set_size, set_axes_equal
except:
    print("Imports for Plot2D File failed.")
    raise


def setupLatexPlot2D(
    figureWidth=483.6969,
    figureHeight=None,
    axisLimX=None,
    axisLimY=None,
    xlabel="$x$",
    ylabel="$y$",
    xTickStep=None,
    yTickStep=None,
):
    if figureHeight is not None:
        width, height = set_size(width=figureWidth, height=figureHeight)
    else:
        width, height = set_size(width=figureWidth)

    fig = plt.figure(figsize=(width, height))
    ax = fig.add_subplot()

    if axisLimX is not None:
        # set axis limits
        ax.set_xlim(axisLimX[0], axisLimX[1])

    if axisLimY is not None:
        ax.set_ylim(axisLimY[0], axisLimY[1])

    # set axis lables
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # set x ticks
    if xTickStep is not None:
        ax.set_xticks(np.arange(axisLimX[0], axisLimX[1] + xTickStep, step=xTickStep))
    if yTickStep is not None:
        ax.set_yticks(np.arange(axisLimY[0], axisLimY[1] + yTickStep, step=yTickStep))
    return fig, ax


def plotImage(rgbImage):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(w, h)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(rgbImage, aspect="auto")


def plotPointSet2D(
    ax,
    X,
    color=[0, 0, 1],
    edgeColor=None,
    size=None,
    markerStyle=None,
    lineWidth=None,
    alpha=None,
    label: str = None,
    zOrder=None,
):
    size = 20 if size is None else size
    markerStyle = "o" if markerStyle is None else markerStyle
    alpha = 1 if alpha is None else alpha
    edgeColor = color if edgeColor is None else edgeColor
    zOrder = 1 if zOrder is None else zOrder
    lineWidth = 1.5 if lineWidth is None else lineWidth

    if label is None:
        ax.scatter(
            X[:, 0],
            X[:, 1],
            color=color,
            alpha=alpha,
            s=size,
            marker=markerStyle,
            edgecolors=edgeColor,
            linewidth=lineWidth,
            zorder=zOrder,
        )
    else:
        ax.scatter(
            X[:, 0],
            X[:, 1],
            color=color,
            label=label,
            alpha=alpha,
            s=size,
            marker=markerStyle,
            edgecolor=None,
            zorder=zOrder,
        )


def plotGraph2D(
    ax,
    X,
    adjacencyMatrix,
    color=None,
    lineColor=None,
    pointSize=None,
    lineWidth=None,
    lineStyle=None,
    pointAlpha=None,
    lineAlpha=None,
    zOrder=None,
):
    color = [0, 0, 1] if color is None else color
    lineColor = [0, 0, 1] if lineColor is None else lineColor
    pointSize = 10 if pointSize is None else pointSize
    lineWidth = 1.5 if lineWidth is None else lineWidth
    pointAlpha = 1 if pointAlpha is None else pointAlpha
    lineAlpha = 1 if lineAlpha is None else lineAlpha
    lineStyle = "-" if lineStyle is None else lineStyle
    zOrder = 1 if zOrder is None else zOrder

    ax.scatter(
        X[:, 0], X[:, 1], color=color, s=pointSize, alpha=pointAlpha, zorder=zOrder
    )
    # check if matrix is symmetric
    if not np.allclose(adjacencyMatrix, adjacencyMatrix.T, rtol=1e-05, atol=1e-08):
        warn("Provided adjacency matrix is not symmetric. Making it symmetric.")
        adjacencyMatrix = 0.5 * (adjacencyMatrix + adjacencyMatrix.T)
    I, J = adjacencyMatrix.shape
    for i in range(0, I):
        for j in range(i, J):
            if adjacencyMatrix[i, j] != 0:
                ax.plot(
                    [X[i, 0], X[j, 0]],
                    [X[i, 1], X[j, 1]],
                    color=lineColor,
                    linewidth=lineWidth,
                    alpha=lineAlpha,
                    linestyle=lineStyle,
                    zorder=zOrder,
                )
    return ax


def plotGraph2_CV(
    rgbImg,
    positions2D,
    adjacencyMatrix,
    lineColor=[0, 81 / 255, 158 / 255],
    circleColor=[0, 81 / 255, 158 / 255],
    lineThickness=5,
    circleRadius=10,
):
    # scale colors
    lineColor = tuple([x * 255 for x in lineColor])
    circleColor = tuple([x * 255 for x in circleColor])

    # draw image
    i = 0
    j = 0
    I, J = adjacencyMatrix.shape
    for i in range(0, I):
        cv2.circle(rgbImg, positions2D[i, :], circleRadius, circleColor, thickness=-1)
        for j in range(0, J):
            if adjacencyMatrix[i, j] == 1:
                cv2.line(
                    rgbImg,
                    (
                        positions2D[i, 0],
                        positions2D[i, 1],
                    ),
                    (
                        positions2D[j, 0],
                        positions2D[j, 1],
                    ),
                    lineColor,
                    lineThickness,
                )
    return rgbImg


def convertColorOpenCV(color):
    return tuple([x * 255 for x in color])


def convertColorsOpenCV(colors):
    convertedColors = []
    for color in colors:
        convertedColors.append(convertColorOpenCV(color))
    return convertedColors


def plotMinimalDistances2D(
    ax,
    X,
    Y,
    correspondanceMatrix=None,
    xColor=None,
    yColor=None,
    correspondanceColor=None,
    xSize=None,
    ySize=None,
    linewidth=None,
    xAlpha=None,
    yAlpha=None,
    lineAlpha=None,
):
    (N, D) = X.shape
    (M, D) = Y.shape
    correspondanceMatrix = (
        np.eye(N, M) if correspondanceMatrix is None else correspondanceMatrix
    )
    xColor = [0, 0, 1] if xColor is None else xColor
    yColor = [1, 0, 0] if yColor is None else yColor
    correspondanceColor = (
        [0.3, 0.3, 0.3] if correspondanceColor is None else correspondanceColor
    )

    for i in range(0, N):
        for j in range(0, M):
            if correspondanceMatrix[i, j] != 0:
                ax.scatter(X[i, 0], X[i, 1], color=xColor, alpha=xAlpha, s=xSize)
                ax.scatter(Y[j, 0], Y[j, 1], color=yColor, alpha=yAlpha, s=ySize)
                ax.plot(
                    [X[i, 0], Y[j, 0]],
                    [X[i, 1], Y[j, 1]],
                    color=correspondanceColor,
                    linewidth=linewidth,
                    alpha=lineAlpha,
                )
    return


def plotCorrespondances2D_CV(
    rgbImg,
    predictionPixelCoordinates: np.array,
    groundTruthPixelCoordinates: np.array,
    predictionColor=None,
    groundTruthColor=None,
    correspondanceColor=None,
    correspondanceLineWidth=None,
    predictionCircleRadius=None,
    groundTruthCircleRadius=None,
    fillMarkers=None,
):
    predictionColor = [0, 0, 1] if predictionColor is None else predictionColor
    groundTruthColor = [1, 0, 1] if groundTruthColor is None else groundTruthColor
    correspondanceColor = (
        [1, 0, 0] if correspondanceColor is None else correspondanceColor
    )
    correspondanceLineWidth = (
        5 if correspondanceLineWidth is None else correspondanceLineWidth
    )
    predictionCircleRadius = (
        10 if predictionCircleRadius is None else predictionCircleRadius
    )
    groundTruthCircleRadius = (
        10 if groundTruthCircleRadius is None else groundTruthCircleRadius
    )
    # convert colors for open cv
    predictionColor, groundTruthColor, correspondanceColor = convertColorsOpenCV(
        [predictionColor, groundTruthColor, correspondanceColor]
    )
    fillMarkers = -1 if fillMarkers is None else fillMarkers

    for predictionPositions2D, groundTruthPositions2D in zip(
        predictionPixelCoordinates, groundTruthPixelCoordinates
    ):
        cv2.line(
            rgbImg,
            (
                predictionPositions2D[0],
                predictionPositions2D[1],
            ),
            (
                groundTruthPositions2D[0],
                groundTruthPositions2D[1],
            ),
            correspondanceColor,
            correspondanceLineWidth,
        )
        cv2.circle(
            rgbImg,
            predictionPositions2D,
            predictionCircleRadius,
            predictionColor,
            fillMarkers,
        )
        cv2.circle(
            rgbImg,
            groundTruthPositions2D,
            groundTruthCircleRadius,
            groundTruthColor,
            fillMarkers,
        )
    return rgbImg
