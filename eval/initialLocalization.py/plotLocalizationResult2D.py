import sys
import os
import matplotlib.pyplot as plt
import numpy as np


try:
    sys.path.append(os.getcwd().replace("/eval", ""))
    from src.evaluation.initialLocalization.initialLocalizationEvaluation import (
        InitialLocalizationEvaluation,
    )
    from src.visualization.plotImg import *
except:
    print("Imports for plotting localization results 2D failed.")
    raise

global eval
eval = InitialLocalizationEvaluation()

resultFolderPaths = ["data/eval/initialLocalization/results/20230807_150735_partial"]
controlOpt = {"resultsToLoad": 0}

styleOpt = {
    "lineColor": [0, 1, 0],
    "lineThickness": 5,
    "circleColor": [0, 81 / 255, 158 / 255],
    "circleRadius": 10,
}


def list_result_files(path):
    return [
        f
        for f in os.listdir(path)
        if (os.path.isfile(os.path.join(path, f)) and f.split(".")[-1] == "pkl")
    ]


if __name__ == "__main__":
    # load results
    resultFolderPath = resultFolderPaths[0]
    resultFiles = list_result_files(resultFolderPath)
    for resultFile in resultFiles:
        resultFilePath = os.path.join(resultFolderPath, resultFile)
        result = eval.loadResults(resultFilePath)
        reprojectionErrorResult = eval.calculateReprojectionError(result)

        # make plot
        # load image
        frame = result["frame"]
        dataSetPath = result["dataSetPath"]
        modelParameters = result["modelParameters"]
        q = result["localizationResult"]["q"]
        model = eval.generateModel(modelParameters)
        model.setGeneralizedCoordinates(q)
        positions3D, adjacencyMatrix = model.getJointPositionsAndAdjacencyMatrix()
        positions2D = eval.reprojectFrom3DRobotBase(positions3D, dataSetPath)
        rgbImg = eval.getDataSet(frame, dataSetPath)[0]  # load image
        lineColor = styleOpt["lineColor"]
        circleColor = styleOpt["circleColor"]
        lineThickness = styleOpt["lineThickness"]
        circleRadius = styleOpt["circleRadius"]
        rgbImg = plotGraphImg(
            rgbImg=rgbImg,
            positions2D=positions2D,
            adjacencyMatrix=adjacencyMatrix,
            lineColor=lineColor,
            circleColor=circleColor,
            lineThickness=lineThickness,
            circleRadius=circleRadius,
        )
        eval.plotImageWithMatplotlib(rgbImg, block=True)

        # save image

        # eval.plotLocalizationResult(
        #     frame,
        #     dataSetPath,
        #     positions3D,
        #     adjacencyMatrix,
        #     lineColor=styleOpt["lineColor"],
        #     circleColor=styleOpt["circleColor"],
        #     lineThickness=styleOpt["lineThickness"],
        #     circleRadius=styleOpt["circleRadius"],
        # )
        # load img
        # plot points in img
        # plot lines in img
        # plot reporjection error in img
    # results = []
    # for resultFilePath in [resultFolderPaths[x] for x in controlOpt["resultsToLoad"]]:
    #     resultFilePath = os.path.join(resultFilePath, resultFileName)
    #     result = eval.,.loadResult(resultFilePath)
    #     results.append(result)
    # # create plot
    # for result in results:
    #     for method in controlOpt["methods"]:
    #         for frame in controlOpt["frames"]:
    #             createPlots(result, frame, method)
