import sys
import os
import matplotlib.pyplot as plt
import numpy as np


try:
    sys.path.append(os.getcwd().replace("/eval", ""))
    from src.evaluation.initialLocalization.initialLocalizationEvaluation import (
        InitialLocalizationEvaluation,
    )
    from src.visualization.plot2D import *
    from src.visualization.colors import *
except:
    print("Imports for plotting localization results 2D failed.")
    raise

global eval
eval = InitialLocalizationEvaluation()

controlOpt = {
    "dataSetsToLoad": [-1],
    "resultsToLoad": [-1],
    "save": True,
    "plotOptions": "multiChrome",  # monoChrome, multiChrome
    "showPlot": False,
    "block": True,
    "saveFolder": "data/eval/initialLocalization/plots/initialLocalizationResults2D",
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
    "lineColor": thesisColors["blue"],
    "lineThickness": 3,
    "circleColor": thesisColors["blue"],
    "circleRadius": 10,
    "colorPalette": thesisColorPalettes["viridis"],
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
    # load results
    for nDataSet, resultFolderPath in enumerate(dataSetsToEvaluate):
        if controlOpt["resultsToLoad"][0] == -1:
            resultFiles = eval.list_result_files(resultFolderPath)
        else:
            resultFiles = eval.list_result_files(resultFolderPath)
            resultFiles = [
                file
                for i, file in enumerate(resultFiles)
                if i in controlOpt["resultsToLoad"]
            ]
        failedFrames = []
        for nResult, resultFile in enumerate(resultFiles):
            resultFilePath = os.path.join(resultFolderPath, resultFile)
            result = eval.loadResults(resultFilePath)
            try:
                # make plot
                lineColor = styleOpt["lineColor"]
                circleColor = styleOpt["circleColor"]
                lineThickness = styleOpt["lineThickness"]
                circleRadius = styleOpt["circleRadius"]
                if controlOpt["plotOptions"] == "monoChrome":
                    rgbImg = eval.plotLocalizationResult2D(
                        result,
                        lineColor,
                        circleColor,
                        lineThickness,
                        circleRadius,
                    )
                elif controlOpt["plotOptions"] == "multiChrome":
                    rgbImg = eval.plotBranchWiseColoredLocalizationResult2D(
                        result,
                        colorPalette=styleOpt["colorPalette"],
                    )
                else:
                    raise NotImplementedError

                # save image
                if controlOpt["save"]:
                    id = "_".join(resultFile.split("_")[0:3])
                    fileName = id + "_" + controlOpt["saveName"]
                    dataSetName = result["dataSetPath"].split("/")[-2]
                    folderPath = os.path.join(controlOpt["saveFolder"], dataSetName)
                    savePath = os.path.join(folderPath, fileName)
                    if not os.path.exists(folderPath):
                        os.makedirs(folderPath, exist_ok=True)
                    eval.saveImage(rgbImg, savePath)
                    if controlOpt["verbose"]:
                        print(
                            "Saved image {}/{} for data set {}/{} initial localization at {}.".format(
                                nResult,
                                len(resultFiles),
                                nDataSet,
                                len(dataSetsToEvaluate),
                                savePath,
                            )
                        )
                if controlOpt["showPlot"]:
                    eval.plotImageWithMatplotlib(rgbImg, block=controlOpt["block"])
                    if not controlOpt["block"]:
                        plt.close("all")
            except:
                failedFrames.append(result["frame"])
        if len(failedFrames) > 0:
            print("Failed on frames {}".format(failedFrames))
