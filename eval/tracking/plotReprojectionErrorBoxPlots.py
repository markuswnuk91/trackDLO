import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib
import pickle
from warnings import warn

try:
    sys.path.append(os.getcwd().replace("/eval", ""))
    from src.evaluation.tracking.trackingEvaluation import TrackingEvaluation
    from src.visualization.plot2D import *
except:
    print("Imports for plotting reprojection errors as boxplot failed.")
    raise

global eval
eval = TrackingEvaluation()

controlOpt = {
    "resultsToLoad": [0, 1, 2],
    "save": True,
    "saveAsTikz": True,
    "showPlot": True,
    "saveFolder": "data/eval/tracking/plots/reprojectionErrorBarPlot",
    "saveName": "reprojectionErrorBarPlot",
    "methodsToEvaluate": ["cpd", "spr", "kpr", "krcpd"],
}
resultFileName = "result.pkl"

resultFolderPaths = [
    "data/eval/tracking/results/20230524_171237_ManipulationSequences_mountedWireHarness_modelY",
    "data/eval/tracking/results/20230807_162939_ManipulationSequences_mountedWireHarness_partial",
    "data/eval/tracking/results/20230524_161235_ManipulationSequences_mountedWireHarness_arena",
]


def loadResult(filePath):
    _, file_extension = os.path.splitext(filePath)
    if file_extension == ".pkl":
        with open(filePath, "rb") as f:
            result = pickle.load(f)
    return result


def createReprojectionErrorBoxPlots(
    dataSetResult,
    methodsToEvaluate=None,
    numSamples=10,
    spacingXTicks=3,
    spacingErrorBars=2,
    capsize=3,
):
    trackingResults = dataSetResult["trackingResults"]
    methodsToEvaluate = (
        list(trackingResults.keys()) if methodsToEvaluate is None else methodsToEvaluate
    )
    availableMethods = [key for key in trackingResults.keys()]

    for method in methodsToEvaluate:
        if method not in availableMethods:
            raise ValueError(
                "Requested method {} not in result file. Provided results are {}. Please include the the necessary information to continue.".format(
                    method, availableMethods
                )
            )
    # compute reprojection errors
    reprojectionErrorEvaluationResults = {}
    for method in methodsToEvaluate:
        reprojectionErrorEvaluationResult = eval.calculateReprojectionErrors(
            trackingResults[method]
        )
        reprojectionErrorEvaluationResults[method] = reprojectionErrorEvaluationResult

    labeledFrames = reprojectionErrorEvaluationResults[methodsToEvaluate[0]][
        "labeledFrames"
    ]

    if numSamples is None:
        framesToPlot = labeledFrames
        sampleIdx = np.array(range(0, len(labeledFrames))).astype(int)
    else:
        sampleIdx = np.round(np.linspace(0, len(labeledFrames) - 1, numSamples)).astype(
            int
        )
        framesToPlot = np.array(labeledFrames)[sampleIdx]

    # Introduce a larger shift between frames
    XTicks = np.arange(len(framesToPlot)) * (
        spacingXTicks + (len(methodsToEvaluate) + 1) * spacingErrorBars
    )
    means = []
    stds = []
    positions = []

    for i, method in enumerate(methodsToEvaluate):
        if method not in availableMethods:
            warn(
                "Could not find requested result for method: {} in evaluation results. Available results are {}. Proceeding without evaluating the requested method.".format(
                    method, availableMethods
                )
            )
            continue
        # Get mean and std results
        means.append(reprojectionErrorEvaluationResults[method]["means"][sampleIdx])
        stds.append(reprojectionErrorEvaluationResults[method]["stds"][sampleIdx])
        # Define the positions for each method's bars
        sign = np.sign((((i + 1) - (len(methodsToEvaluate) + 1) / 2)))
        shift = (
            sign
            * np.abs(((i + 1) - (len(methodsToEvaluate) + 1) / 2))
            * spacingErrorBars
        )
        position = XTicks + shift
        positions.append(position)

    # plot error bars
    for i, method in enumerate(methodsToEvaluate):
        plt.errorbar(
            positions[i],
            means[i],
            yerr=stds[i],
            fmt="o",
            label=method,
            capsize=capsize,
        )

    # Set the title, labels, and a legend
    plt.xlabel("frames")
    plt.ylabel("Results")
    plt.xticks(XTicks, framesToPlot)
    plt.legend()
    plt.tight_layout()

    if controlOpt["save"]:
        # make folder for dataSet
        dataSetPath = dataSetResult["dataSetPath"]
        dataSetName = dataSetPath.split("/")[-2]
        saveFolderPath = os.path.join(controlOpt["saveFolder"], dataSetName)
        if not os.path.exists(saveFolderPath):
            os.makedirs(saveFolderPath)
        fileName = controlOpt["saveName"]
        savePath = os.path.join(saveFolderPath, fileName)
        # save as png
        plt.savefig(savePath)
        # save as tixfigure
        if controlOpt["saveAsTikz"]:
            tikzplotlib.save(savePath + ".tex")
    if controlOpt["showPlot"]:
        plt.show(block=True)
    return


if __name__ == "__main__":
    # load all results
    dataSetResults = []
    for resultFilePath in [resultFolderPaths[x] for x in controlOpt["resultsToLoad"]]:
        resultFilePath = os.path.join(resultFilePath, resultFileName)
        result = loadResult(resultFilePath)
        dataSetResults.append(result)
    # create plot
    for dataSetResult in dataSetResults:
        createReprojectionErrorBoxPlots(
            dataSetResult, methodsToEvaluate=controlOpt["methodsToEvaluate"]
        )
