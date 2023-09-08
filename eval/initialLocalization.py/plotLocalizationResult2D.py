import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import pickle


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

resultFolderPaths = ["data/eval/initialLocalization/results/20230807_150735_partial"]
controlOpt = {"resultsToLoad": 0}


def list_files(path):
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]


if __name__ == "__main__":
    # load results
    resultFolderPath = resultFolderPaths[0]
    resultFiles = list_files(resultFolderPath)
    for resultFile in resultFiles:
        resultFilePath = os.path.join(resultFolderPath, resultFile)
        result = eval.loadResults(resultFilePath)
        reprojectionErrorResult = eval.evaluateReprojectionError(result)

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
