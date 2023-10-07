import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from collections import defaultdict
from scipy.optimize import curve_fit
from scipy.stats import norm

try:
    sys.path.append(os.getcwd().replace("/eval", ""))
    from src.evaluation.graspingAccuracy.graspingAccuracyEvaluation import (
        GraspingAccuracyEvaluation,
    )
    from src.visualization.plot2D import *
    from src.visualization.colors import *
except:
    print("Imports for plotting script tracking error time series failed.")
    raise

global eval
eval = GraspingAccuracyEvaluation()

controlOpt = {
    "resultsToLoad": [-1],
    "methodsToEvaluate": ["cpd", "spr", "kpr"],
    "registrationResultsToEvaluate": [-1],
    "showPlot": True,
    "makeScatterPlot": False,
    "save": True,
    "verbose": True,
}

saveOpt = {
    "saveFolder": "data/eval/graspingAccuracy/plots/graspingPredictionResult",
    "saveFileName": "graspingPrediction",
}

styleOpt = {
    "methodColors": {
        "cpd": thesisColors["susieluMagenta"],
        "spr": thesisColors["susieluGold"],
        "kpr": thesisColors["susieluBlue"],
    },
    "modelMarkers": {
        "modelY": "o",
        "partial": "o",  # "s"
        "arena": "o",  # "^"
    },
    "alpha": 0.7,
    "markersize": 20,
    "legendMarkerSize": 5,
    "translationalErrorThreshold": 0.05,  # None: do not plot
    "rotationalErrorThreshold": 45,  # None: do not plot
    "translationalThresholdLineColor": [1, 0, 0],
    "rotationalThresholdLineColor": [1, 0, 0],
    "thresholdLineStyle": "--",
}

resultFileName = "result.pkl"
resultFolderPaths = [
    "data/eval/graspingAccuracy/results/20230522_130903_modelY",
    "data/eval/graspingAccuracy/results/20230522_131545_modelY",
    "data/eval/graspingAccuracy/results/20230522_154903_modelY",
    "data/eval/graspingAccuracy/results/20230807_142319_partial",
    "data/eval/graspingAccuracy/results/20230807_142909_partial",
    "data/eval/graspingAccuracy/results/20230807_143737_partial",
    "data/eval/graspingAccuracy/results/20230522_140014_arena",
    "data/eval/graspingAccuracy/results/20230522_141025_arena",
    "data/eval/graspingAccuracy/results/20230522_142058_arena",
]


def evaluateGraspingAccuracy(
    result,
    method,
    num,
):
    registrationResult = result["trackingResults"][method]["registrationResults"][num]
    frame = registrationResult["frame"]
    dataSetPath = result["dataSetPath"]

    # ground truth
    (
        groundTruthGraspingPose,
        groundTruthGraspingPosition,
        groundTruthGraspingRotationMatrix,
    ) = eval.loadGroundTruthGraspingPose(
        dataSetPath, frame + 1
    )  # ground truth grasping position is given by the frame after the prediction frame
    groundTruthGraspingAxis = groundTruthGraspingRotationMatrix[:3, 0]
    # prediction
    graspingLocalCoordinates = eval.loadGraspingLocalCoordinates(dataSetPath)
    graspingLocalCoordinate = graspingLocalCoordinates[num]
    T = registrationResult["result"]["T"]
    B = result["trackingResults"][method]["B"]
    S = result["initializationResult"]["localizationResult"]["SInit"]
    (
        predictedGraspingPosition,
        predictedGraspingAxis,
    ) = eval.predictGraspingPositionAndAxisFromRegistrationTargets(
        T, B, S, graspingLocalCoordinate
    )
    graspingAccuracy = eval.calculateGraspingAccuracyError(
        predictedGraspingPosition=predictedGraspingPosition,
        predictedGraspingAxis=predictedGraspingAxis,
        groundTruthGraspingPose=groundTruthGraspingPose,
    )
    return graspingAccuracy


def scatterPlotGraspingErrors(
    translationalGraspingErrors,
    rotationalGraspingErrors,
    correspondingMethods,
    correspondingModelNames,
    colors=None,
    markers=None,
    alpha=0.3,
    translationalThreshold=None,
    rotationalThreshold=None,
    translationalThresholdLineColor=None,
    rotationalThresholdLineColor=None,
    thresholdLineStyle=None,
):
    fig, ax = setupLatexPlot2D()
    if colors is None:
        colors = []
        for method in correspondingMethods:
            if method == "cpd":
                colors.append([1, 0, 0, alpha])
            elif method == "spr":
                colors.append([0, 0, 1, alpha])
            elif method == "kpr":
                colors.append([0, 1, 0, alpha])
            elif method == "krcpd":
                colors.append([1, 1, 0, alpha])
            else:
                colors.append([0.7, 0.7, 0.7, alpha])
    if markers is None:
        markers = []
        for model in correspondingModelNames:
            if model == "modelY":
                markers.append("o")
            elif model == "partial":
                markers.append("^")
            elif model == "arena":
                markers.append("s")
            elif model == "singleDLO":
                markers.append("D")

    translationalThresholdLineColor = (
        [1, 0, 0]
        if translationalThresholdLineColor is None
        else translationalThresholdLineColor
    )
    rotationalThresholdLineColor = (
        [1, 0, 0]
        if rotationalThresholdLineColor is None
        else rotationalThresholdLineColor
    )
    thresholdLineStyle = "-" if thresholdLineStyle is None else thresholdLineStyle

    for i, (transplationalError, rotationalError) in enumerate(
        zip(translationalGraspingErrors, rotationalGraspingErrors)
    ):
        ax.scatter(
            transplationalError,
            rotationalError,
            color=colors[i],
            marker=markers[i],
            alpha=alpha,
            s=styleOpt["markersize"],
        )
    # create legend
    methodsToList = list(set(correspondingMethods))
    legendSymbols = []
    for label in methodsToList:
        legendSymbol = Line2D(
            [],
            [],
            marker=markers[correspondingMethods.index(label)],
            color=colors[correspondingMethods.index(label)],
            linestyle="None",
            label=label,
            markersize=styleOpt["legendMarkerSize"],
        )
        legendSymbols.append(legendSymbol)
    ax.legend(handles=legendSymbols)
    # threshold
    if translationalThreshold is not None and rotationalThreshold is not None:
        plt.axvline(
            x=translationalThreshold,
            ymin=0,
            ymax=(rotationalThreshold - ax.get_ylim()[0])
            / (ax.get_ylim()[1] - ax.get_ylim()[0]),
            color=translationalThresholdLineColor,
            linestyle=thresholdLineStyle,
        )
        plt.axhline(
            y=rotationalThreshold,
            xmin=0,
            xmax=(translationalThreshold - ax.get_xlim()[0])
            / (ax.get_xlim()[1] - ax.get_xlim()[0]),
            color=rotationalThresholdLineColor,
            linestyle=thresholdLineStyle,
        )
    return fig, ax


def graspingErrorsHistogram(
    translationalGraspingErrors,
    correspondingMethods,
    correspondingModelNames,
    n_bins=20,
):
    # Ensure that the two lists have the same length
    assert len(translationalGraspingErrors) == len(correspondingMethods)

    # Create a defaultdict with lists as default values
    grouped_vals = defaultdict(list)

    # Iterate through both lists simultaneously using zip
    for method, val in zip(correspondingMethods, translationalGraspingErrors):
        grouped_vals[method].append(val)

    # Convert defaultdict to a regular dict (optional)
    grouped_vals = dict(grouped_vals)
    cols = []
    histogramColors = []
    for key in grouped_vals:
        cols.append(grouped_vals[key])
        histogramColors.append(styleOpt["methodColors"][key])
    x = np.vstack((cols)).T

    # x = x - np.mean(x, axis=0)
    fig, ax = setupLatexPlot2D()
    hist_handle = ax.hist(
        x, n_bins, density=False, histtype="bar", color=histogramColors
    )

    # # plot gaussian
    # x_axis = np.linspace(hist_handle[1][0], hist_handle[1][-1], 1000)
    # fitLineStyle = "--"
    # for method in grouped_vals:
    #     mu = np.mean(grouped_vals[method])
    #     std = np.std(grouped_vals[method])
    #     p = norm.pdf(x_axis, mu, std)
    #     ax.plot(
    #         x_axis, p, color=styleOpt["methodColors"][method], linestyle=fitLineStyle
    #     )

    # create legend
    legendSymbols = []
    for method in grouped_vals:
        legendSymbol = Patch(
            facecolor=styleOpt["methodColors"][method],
            label=method,
        )
        legendSymbols.append(legendSymbol)
    # for method in grouped_vals:
    #     legendSymbol = Line2D(
    #         [],
    #         [],
    #         color=styleOpt["methodColors"][method],
    #         linestyle=fitLineStyle,
    #         label="fitted gaussian, " + method,
    #     )
    #     legendSymbols.append(legendSymbol)
    ax.legend(handles=legendSymbols)
    return fig, ax


if __name__ == "__main__":
    if controlOpt["resultsToLoad"][0] == -1:
        resultsToEvaluate = resultFolderPaths
    else:
        resultsToEvaluate = [
            resultFolderPath
            for i, resultFolderPath in enumerate(resultFolderPaths)
            if i in controlOpt["resultsToLoad"]
        ]

    graspingAccuracyResults = []
    translationalGraspingErrors = []
    rotationalGraspingErrors = []
    methods = []
    dataSets = []
    models = []
    grasps = []
    plotColors = []
    plotMarkers = []
    for nResult, resultFolderPath in enumerate(resultsToEvaluate):
        resultFilePath = os.path.join(resultFolderPath, resultFileName)
        result = eval.loadResults(resultFilePath)

        existingMethods = eval.getRegistrationMethods(result)
        methodsToEvaluate = [
            method
            for method in existingMethods
            if method in controlOpt["methodsToEvaluate"]
        ]
        for nMethod, method in enumerate(methodsToEvaluate):
            numRegistrationResults = eval.getNumRegistrationResults(result)
            if controlOpt["registrationResultsToEvaluate"][0] == -1:
                registrationResultsToEvaluate = list(
                    range(
                        0, numRegistrationResults - 1
                    )  # do not evaluate last registration result since this is only the final frame
                )
            else:
                registrationResultsToEvaluate = controlOpt[
                    "registrationResultsToEvaluate"
                ]
            for nRegistrationResult in registrationResultsToEvaluate:
                graspingAccuracyError = evaluateGraspingAccuracy(
                    result, method, nRegistrationResult
                )
                graspingAccuracyResults.append(graspingAccuracyError)
                translationalGraspingErrors.append(
                    graspingAccuracyError["graspingPositionErrors"]
                )
                rotationalGraspingErrors.append(
                    graspingAccuracyError["graspingAngularErrorsInGrad"]
                )
                methods.append(method)
                grasps.append(nRegistrationResult)
                modelName = result["dataSetName"].split("_")[-1]
                models.append(modelName)
                plotColors.append(styleOpt["methodColors"][method])
                plotMarkers.append(styleOpt["modelMarkers"][modelName])
    # if controlOpt["showPlot"]:

    # if controlOpt["save"]:
    #     dataSetName = result["dataSetName"]
    #     fileID = "_".join(
    #         result["trackingResults"][method]["registrationResults"][
    #             nRegistrationResult
    #         ]["fileName"].split("_")[0:3]
    #     )
    #     fileName = fileID + "_" + saveOpt["saveFileName"]
    #     saveFolderPath = saveOpt["saveFolder"]
    #     saveFolderPath = os.path.join(saveFolderPath, dataSetName, method)
    #     saveFilePath = os.path.join(saveFolderPath, fileName)
    #     if not os.path.exists(saveFolderPath):
    #         os.makedirs(saveFolderPath, exist_ok=True)
    #     eval.saveImage(rgbImg, saveFilePath)
    #     if controlOpt["verbose"]:
    #         print(
    #             "Saved registration {}/{} from method {}/{} of result {}/{} at {}".format(
    #                 nRegistrationResult + 1,
    #                 len(registrationResultsToEvaluate),
    #                 nMethod + 1,
    #                 len(methodsToEvaluate),
    #                 nResult + 1,
    #                 len(resultsToEvaluate),
    #                 saveFilePath,
    #             )
    #         )
    meanTranslationalErrors = np.mean(translationalGraspingErrors)
    stdTranslationalErrors = np.std(translationalGraspingErrors)
    meanRotationalErrors = np.mean(rotationalGraspingErrors)
    stdRotationalErrors = np.std(rotationalGraspingErrors)
    if controlOpt["makeScatterPlot"]:
        fig_scatterPlot, ax_scatterPlot = scatterPlotGraspingErrors(
            translationalGraspingErrors=translationalGraspingErrors,
            rotationalGraspingErrors=rotationalGraspingErrors,
            correspondingMethods=methods,
            correspondingModelNames=models,
            colors=plotColors,
            markers=plotMarkers,
            alpha=styleOpt["alpha"],
            translationalThreshold=styleOpt["translationalErrorThreshold"],
            rotationalThreshold=styleOpt["rotationalErrorThreshold"],
            translationalThresholdLineColor=styleOpt["translationalThresholdLineColor"],
            rotationalThresholdLineColor=styleOpt["rotationalThresholdLineColor"],
            thresholdLineStyle=styleOpt["thresholdLineStyle"],
        )
        if controlOpt["showPlot"]:
            plt.show(block=True)

    fig_histogram, ax_histogram = graspingErrorsHistogram(
        translationalGraspingErrors=translationalGraspingErrors,
        correspondingMethods=methods,
        correspondingModelNames=models,
    )
    plt.show(block=True)
    if controlOpt["verbose"]:
        print("Finished result generation.")
