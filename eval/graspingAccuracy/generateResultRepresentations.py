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

# script control parameters
controlOptions = {
    "printResultTables": False,
    "plot2DGraspingError": True,
    "scatterPlotGraspingErrors": False,
}

global eval
pathToConfigFile = (
    os.path.dirname(os.path.abspath(__file__))
    + "/evalConfigs/representationConfig.json"
)
eval = GraspingAccuracyEvaluation(configFilePath=pathToConfigFile)

resultRootFolderPath = "data/eval/graspingAccuracy/results"
resultFileName = "result"


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


def accumulateGraspingErrors(results):
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
    return (
        translationalGraspingErrors,
        rotationalGraspingErrors,
        correspondingMethods,
        correspondingModelNames,
    )


def scatterPlotGraspingErrors(results):
    alpha = 0.3
    (
        translationalGraspingErrors,
        rotationalGraspingErrors,
        correspondingMethods,
        correspondingModelNames,
    ) = accumulateGraspingErrors(results)

    colors = []
    for method in correspondingMethods:
        if method == "cpd":
            colors.append([1, 0, 0, alpha])
        elif method == "spr":
            colors.append([0, 0, 1, alpha])
        elif method == "krcpd":
            colors.append([0, 1, 0, alpha])
        elif method == "krcpd4BDLO":
            colors.append([1, 1, 0, alpha])
        else:
            colors.append([0.7, 0.7, 0.7, alpha])
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


def visualizeGraspingError2D(graspingAccuracyResult):
    groundTruthColor = [0, 1, 0]
    predictionColor = [1, 0, 0]
    method = graspingAccuracyResult["method"]
    dataSetName = graspingAccuracyResult["dataSetPath"].split("/")[-2]
    frames = graspingAccuracyResult["predictedFrames"]
    dataSetPath = graspingAccuracyResult["dataSetPath"]
    groundTruthGraspingPositions3D = np.array(
        graspingAccuracyResult["graspingPositions"]["groundTruth"]
    )
    predictedGraspingPositions3D = np.array(
        graspingAccuracyResult["graspingPositions"]["predicted"]
    )
    groundTruthGraspingAxes3D = np.array(
        graspingAccuracyResult["graspingAxes"]["groundTruth"]
    )
    predictedGraspingAxes3D = np.array(
        graspingAccuracyResult["graspingAxes"]["predicted"]
    )
    rgbImages = []
    for i, frame in enumerate(frames):
        graspingPoses = np.vstack(
            (groundTruthGraspingPositions3D[i], predictedGraspingPositions3D[i])
        )
        graspingAxes = np.vstack(
            (groundTruthGraspingAxes3D[i], predictedGraspingAxes3D[i])
        )
        colors = [groundTruthColor, predictionColor]
        rgbImages.append(
            eval.visualizeGraspingPoses2D(
                frame, dataSetPath, graspingPoses, graspingAxes, colors
            )
        )

    for i, rgbImage in enumerate(rgbImages):
        savePath = eval.config["resultRootFolderPath"] + "/imgs/graspingResults/"
        fileName = dataSetName + "_" + method + "_" "graspingPose_" + str(i)
        eval.plotImageWithMatplotlib(
            rgbImage=rgbImage,
            show=True,
            block=True,
            save=False,
            savePath=savePath,
            fileName=fileName,
        )
    plt.show(block=False)
    return


def tabularizeResults(
    results,
    methodsToPrint=["cpd", "spr", "krcpd4BDLO"],
    modelsToPrint=["modelY", "partialWireHarness", "arenaWireHarness"],
    resultScaleFactor=100,
):
    (
        translationalGraspingErrors,
        rotationalGraspingErrors,
        correspondingMethods,
        correspondingModels,
    ) = accumulateGraspingErrors(results)

    evaluationResults = {
        "nGrasps": {},
        "translationalError": {},
        "translationalStdDev": {},
        "rotationalError": {},
        "rotationalStdDev": {},
    }
    models = list(set(correspondingModels))
    methods = list(set(correspondingMethods))

    for i, model in enumerate(models):
        evaluationResults["nGrasps"][model] = int(
            correspondingModels.count(model) / len(methods)
        )
        evaluationResults["translationalError"][model] = {}
        evaluationResults["translationalStdDev"][model] = {}
        evaluationResults["rotationalError"][model] = {}
        evaluationResults["rotationalStdDev"][model] = {}
        for j, method in enumerate(methods):
            # gather all errors for the combination of model and method
            translationalIndices = [
                index
                for index, value in enumerate(translationalGraspingErrors)
                if correspondingModels[index] == model
                and correspondingMethods[index] == method
            ]
            evaluationResults["translationalError"][model][method] = np.mean(
                np.array(translationalGraspingErrors)[translationalIndices]
            )
            evaluationResults["translationalStdDev"][model][method] = np.std(
                np.array(translationalGraspingErrors)[translationalIndices]
            )

            rotationalIndices = [
                index
                for index, value in enumerate(rotationalGraspingErrors)
                if correspondingModels[index] == model
                and correspondingMethods[index] == method
            ]
            evaluationResults["rotationalError"][model][method] = np.mean(
                np.array(rotationalGraspingErrors)[rotationalIndices]
            )
            evaluationResults["rotationalStdDev"][model][method] = np.std(
                np.array(rotationalGraspingErrors)[rotationalIndices]
            )

    # generate tabels
    # translational
    translational_tabel = r"""translational result table:
    ----------Cut-------------
    """
    translationalTabelResults = []
    for model in modelsToPrint:
        resultRow = []
        resultRow.append(evaluationResults["nGrasps"][model])
        for method in methodsToPrint:
            resultRow.append(evaluationResults["translationalError"][model][method])
            resultRow.append(evaluationResults["translationalStdDev"][model][method])
        translationalTabelResults.append(resultRow)
    for model, tableRow in zip(modelsToPrint, translationalTabelResults):
        modelID = eval.getModelID(model)
        nGrasps = tableRow[0]
        formatted_table_row = [
            format(r * resultScaleFactor, ".1f") for r in tableRow[1:]
        ]
        # Identify the lowest mean
        min_mean = min(formatted_table_row[2::2], key=float)
        min_std = min(formatted_table_row[3::2], key=float)
        # Wrap the lowest mean with \textbf{}
        formatted_table_row[2::2] = [
            r if r != min_mean else f"\\textbf{{{r}}}"
            for r in formatted_table_row[2::2]
        ]
        # Wrap the lowest std with \textbf{}
        formatted_table_row[3::2] = [
            r if r != min_std else f"\\textbf{{{r}}}" for r in formatted_table_row[3::2]
        ]
        translational_tabel += f"\t{modelID} & {nGrasps} & {' & '.join(map(str, formatted_table_row))}\\\\\n"
    translational_tabel += r"""----------Cut-------------
    """
    # rotational
    rotational_tabel = r"""rotational result table:
    ----------Cut-------------
    """
    rotationalTableResults = []
    for model in modelsToPrint:
        resultRow = []
        resultRow.append(evaluationResults["nGrasps"][model])
        for method in methodsToPrint:
            resultRow.append(evaluationResults["rotationalError"][model][method])
            resultRow.append(evaluationResults["rotationalStdDev"][model][method])
        rotationalTableResults.append(resultRow)
    for model, tableRow in zip(modelsToPrint, rotationalTableResults):
        modelID = eval.getModelID(model)
        nGrasps = tableRow[0]
        formatted_table_row = [format(r, ".1f") for r in tableRow[1:]]
        # Identify the lowest mean
        min_mean = min(formatted_table_row[2::2], key=float)
        min_std = min(formatted_table_row[3::2], key=float)
        # Wrap the lowest mean with \textbf{}
        formatted_table_row[2::2] = [
            r if r != min_mean else f"\\textbf{{{r}}}"
            for r in formatted_table_row[2::2]
        ]
        # Wrap the lowest std with \textbf{}
        formatted_table_row[3::2] = [
            r if r != min_std else f"\\textbf{{{r}}}" for r in formatted_table_row[3::2]
        ]
        rotational_tabel += f"\t{modelID} & {nGrasps} & {' & '.join(map(str, formatted_table_row))}\\\\\n"
    rotational_tabel += r"""----------Cut-------------
    """
    return translational_tabel, rotational_tabel


if __name__ == "__main__":
    # load all results
    results = loadResults(resultRootFolderPath)
    if controlOptions["printResultTables"]:
        print(tabularizeResults(results)[0])
        print(tabularizeResults(results)[1])

    if controlOptions["plot2DGraspingError"]:
        for i, dataSetResult in enumerate(results):
            for j, graspingAccuracyResult in enumerate(
                dataSetResult["graspingAccuracyResults"]
            ):
                visualizeGraspingError2D(
                    graspingAccuracyResult=results[i]["graspingAccuracyResults"][j]
                )

    # plot result representations
    if controlOptions["scatterPlotGraspingErrors"]:
        scatterPlotGraspingErrors(results)

    # save result representations

    print("Done.")
