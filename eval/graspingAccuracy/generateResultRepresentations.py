import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from warnings import warn

try:
    sys.path.append(os.getcwd().replace("/eval", ""))
    from src.evaluation.graspingAccuracy.graspingAccuracyEvaluation import (
        GraspingAccuracyEvaluation,
    )
except:
    print("Imports for Grasping Accuracy Result Evaluation failed.")
    raise
global eval
pathToConfigFile = (
    os.path.dirname(os.path.abspath(__file__))
    + "/evalConfigs/representationConfig.json"
)
eval = GraspingAccuracyEvaluation(configFilePath=pathToConfigFile)

resultRootFolderPath = eval.config["resultRootFolderPath"]
resultFileName = eval.config["resultFileName"]


def loadResults(resultRootFolderPath):
    results = []
    subfolders = [
        f
        for f in os.listdir(resultRootFolderPath)
        if os.path.isdir(os.path.join(resultRootFolderPath, f))
    ]
    for subfolder in subfolders:
        resultFolderPath = os.path.join(resultRootFolderPath, subfolder)
        resultFileNames = [
            f
            for f in os.listdir(resultFolderPath)
            if os.path.isfile(os.path.join(resultFolderPath, f)) and resultFileName in f
        ]
        if len(resultFileNames) > 1:
            warn(
                "Multiple result files found in folder {}. Using only the first one.".format(
                    resultFolderPath
                )
            )
        resultFilePath = os.path.join(resultFolderPath, resultFileNames[0])
        result = eval.loadResults(resultFilePath)
        results.append(result)

    return results


def scatterPlotGraspingErrors(results):
    alpha = 0.3
    translationalGraspingErrors = []
    rotationalGraspingErrors = []
    correspondingMethods = []
    correspondingModelNames = []
    for dataSetIndex, result in enumerate(results):
        for registrationMethodIndex, registrationMethodResult in enumerate(
            result["graspingAccuracyResults"]
        ):
            registrationMethod = registrationMethodResult["method"]
            numGraspingPositions = len(
                registrationMethodResult["graspingPositionErrors"]
            )
            for graspingIndex in range(0, numGraspingPositions):
                # get translational grasping error
                translationalGraspingError = results[dataSetIndex][
                    "graspingAccuracyResults"
                ][registrationMethodIndex]["graspingPositionErrors"][graspingIndex]
                translationalGraspingErrors.append(translationalGraspingError)

                if translationalGraspingError > 1.5:
                    print("Here")
                # get rotational grasping error
                rotationalGraspingError = results[dataSetIndex][
                    "graspingAccuracyResults"
                ][registrationMethodIndex]["graspingAngularErrors"]["rad"][
                    graspingIndex
                ]
                rotationalGraspingErrors.append(rotationalGraspingError)

                # get model type
                modelName = results[dataSetIndex]["graspingAccuracyResults"][
                    registrationMethodIndex
                ]["trackingResult"]["initializationResult"]["modelParameters"][
                    "modelInfo"
                ][
                    "name"
                ]
                correspondingModelNames.append(modelName)

                # get corresponding method
                correspondingMethods.append(registrationMethod)

    colors = []
    for method in correspondingMethods:
        if method == "cpd":
            colors.append([1, 0, 0])
        elif method == "spr":
            colors.append([0, 0, 1])
        elif method == "krcpd":
            colors.append([0, 1, 0])
        elif method == "krcpd4BDLO":
            colors.append([1, 1, 0])
        else:
            colors.append([0.7, 0.7, 0.7])
    # for model in correspondingModelNames:
    #     if model == "modelY":
    #         colors.append([1, 0, 0, alpha])
    #     elif model == "partialWireHarness":
    #         colors.append([0, 0, 1, alpha])
    #     elif model == "arenaWireHarness":
    #         colors.append([0, 1, 0, alpha])
    #     elif model == "singleDLO":
    #         colors.append([1, 1, 0, alpha])
    #     else:
    #         colors.append([0.7, 0.7, 0.7, 0.1])

    markers = []
    # for method in correspondingMethods:
    #     if method == "cpd":
    #         markers.append("^")
    #     elif method == "spr":
    #         markers.append("s")
    #     elif method == "krcpd":
    #         markers.append("o")
    #     elif method == "krcpd4BDLO":
    #         markers.append("D")
    #     else:
    #         markers.append(".")
    for model in correspondingModelNames:
        if model == "modelY":
            markers.append("o")
        elif model == "partialWireHarness":
            markers.append("^")
        elif model == "arenaWireHarness":
            markers.append("s")
        elif model == "singleDLO":
            markers.append("D")
        else:
            colors.append([0.7, 0.7, 0.7, 0.1])
    for i, marker in enumerate(markers):
        plt.scatter(
            rotationalGraspingErrors[i],
            translationalGraspingErrors[i],
            c=colors[i],
            marker=marker,
        )
    plt.show(block=True)
    return


def tabularizeResults(results):
    return


if __name__ == "__main__":
    # load all results
    results = loadResults(resultRootFolderPath)

    # plot result representations
    scatterPlotGraspingErrors(results)
    # tabularize results

    # save result representations

    print("Done.")
