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

controlOpt = {
    "dataSetsToLoad": [2],
    # "resultsToLoad": [1],
    "save": True,
    "showPlot": False,
    "block": False,
    "saveFolder": "data/eval/initialLocalization/plots/initialLocalizationResults",
    "saveName": "initialLocalizationResultImg",
    "verbose": True,
}

resultFolderPaths = [
    "data/eval/initialLocalization/results/20230603_143937_modelY",
    "data/eval/initialLocalization/results/20230516_112207_YShape",
    "data/eval/initialLocalization/results/20230807_150735_partial",
    "data/eval/initialLocalization/results/20230516_113957_Partial",
    "data/eval/initialLocalization/results/20230516_115857_arena",
]

styleOpt = {
    "lineColor": [0, 1, 0],
    "lineThickness": 5,
    "circleColor": [0, 81 / 255, 158 / 255],
    "circleRadius": 10,
}

if __name__ == "__main__":
    if controlOpt["dataSetsToLoad"][0] == -1:
        dataSetsToEvaluate = resultFolderPaths
    else:
        dataSetsToEvaluate = [
            dataSetPath
            for i, dataSetPath in enumerate(resultFolderPaths)
            if i in controlOpt["dataSetsToLoad"]
        ]
    for resultFolderPath in dataSetsToEvaluate:
        # load results
        resultFiles = eval.list_result_files(resultFolderPath)
        failedFrames = []

        for resultFile in resultFiles:
            resultFilePath = os.path.join(resultFolderPath, resultFile)
            result = eval.loadResults(resultFilePath)
            try:
                # make plot
                # load image
                frame = result["frame"]
                dataSetPath = result["dataSetPath"]
                modelParameters = result["modelParameters"]
                q = result["localizationResult"]["q"]
                model = eval.generateModel(modelParameters)
                model.setGeneralizedCoordinates(q)
                (
                    positions3D,
                    adjacencyMatrix,
                ) = model.getJointPositionsAndAdjacencyMatrix()
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
                if controlOpt["showPlot"]:
                    eval.plotImageWithMatplotlib(rgbImg, block=controlOpt["block"])
                    if not controlOpt["block"]:
                        plt.close("all")
                # save image
                if controlOpt["save"]:
                    id = "_".join(resultFile.split("_")[0:3])
                    fileName = id + "_" + controlOpt["saveName"]
                    dataSetName = dataSetPath.split("/")[-2]
                    folderPath = os.path.join(controlOpt["saveFolder"], dataSetName)
                    savePath = os.path.join(folderPath, fileName)
                    if not os.path.exists(folderPath):
                        os.makedirs(folderPath, exist_ok=True)
                    eval.saveImage(rgbImg, savePath)
                if controlOpt["verbose"]:
                    print(
                        "Saved image for initial localization result {} for at {}.".format(
                            resultFile, savePath
                        )
                    )
            except:
                failedFrames.append(result["frame"])
        if len(failedFrames) > 0:
            print("Failed on frames {}".format(failedFrames))
