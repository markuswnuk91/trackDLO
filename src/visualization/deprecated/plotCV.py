import sys, os
import cv2

try:
    sys.path.append(os.getcwd().replace("/src/visualization", ""))
except:
    print("Imports for plot image fuctions file failed.")
    raise


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
