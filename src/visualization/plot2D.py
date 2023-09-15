import sys, os
import matplotlib.pyplot as plt
import numpy as np
import cv2

try:
    sys.path.append(os.getcwd().replace("/src/visualization", ""))
    from src.visualization.plotUtils import set_size, set_axes_equal
except:
    print("Imports for Plot2D File failed.")
    raise


def setupLatexPlot2D(
    figureWidth=483.6969,
    figureHeight=None,
    axisLimX=[0, 1],
    axisLimY=[0, 1],
    xlabel="$x$",
    ylabel="$y$",
    xTickStep=None,
    yTickStep=None,
):
    if figureHeight is not None:
        fig = plt.figure(figsize=set_size(width=figureWidth, height=figureHeight))
    else:
        fig = plt.figure(figsize=set_size(width=figureWidth))
    ax = fig.add_subplot()

    # set axis limits
    ax.set_xlim(axisLimX[0], axisLimX[1])
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


def plotGraph2D(
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


def plotCorrespondances2D(
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
