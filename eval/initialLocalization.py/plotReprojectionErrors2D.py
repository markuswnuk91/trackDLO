import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import traceback

try:
    sys.path.append(os.getcwd().replace("/eval", ""))
    from src.evaluation.initialLocalization.initialLocalizationEvaluation import (
        InitialLocalizationEvaluation,
    )
    from src.visualization.plot2D import *
except:
    print("Imports for plotting localization results 2D failed.")
    raise

global eval
eval = InitialLocalizationEvaluation()

controlOpt = {
    "dataSet": 4,
    "frames": [1],
    "save": False,
    "showPlot": True,
    "block": True,
    "saveFolder": "data/eval/initialLocalization/plots/reprojectionErrors2D",
    "saveName": "initialLocalizationResultImg",
    "verbose": True,
}

resultFolderPaths = [
    "data/eval/initialLocalization/results/20230603_143937_modelY",
    "data/eval/initialLocalization/results/20230516_112207_YShape",
    "data/eval/initialLocalization/results/20230807_150735_partial",
    "data/eval/initialLocalization/results/20230516_113957_Partial",
    "data/eval/initialLocalization/results/20230516_115857_arena",
    "data/eval/initialLocalization/results/20230603_140143_arena",
]

styleOpt = {
    "lineColor": [0, 1, 0],
    "lineThickness": 5,
    "circleColor": [0, 81 / 255, 158 / 255],
    "circleRadius": 10,
}

if __name__ == "__main__":
    resultFolderPath = resultFolderPaths[controlOpt["dataSet"]]
    # load results
    resultFiles = eval.list_result_files(resultFolderPath)
    failedFrames = []
    for resultFile in resultFiles:
        resultFilePath = os.path.join(resultFolderPath, resultFile)
        result = eval.loadResults(resultFilePath)
        # try:
        # make plot
        lineColor = styleOpt["lineColor"]
        circleColor = styleOpt["circleColor"]
        lineThickness = styleOpt["lineThickness"]
        circleRadius = styleOpt["circleRadius"]
        rgbImg = eval.plotReprojectionErrors2D(
            result,
        )
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
        if controlOpt["showPlot"]:
            eval.plotImageWithMatplotlib(rgbImg, block=controlOpt["block"])
            if not controlOpt["block"]:
                plt.close("all")
        # except:
        #     failedFrames.append(result["frame"])
        #     traceback.print_exc()
    if len(failedFrames) > 0:
        print("Failed on frames {}".format(failedFrames))