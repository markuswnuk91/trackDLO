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
    from src.visualization.colors import *
    from src.visualization.plotUtils import *
except:
    print("Imports for plotting reprojection errors as boxplot failed.")
    raise

global eval
eval = TrackingEvaluation()

controlOpt = {
    "resultsToLoad": [0],  # 0,1,2
    "save": True,
    "saveAs": "pdf",  # "pdf" "tikz" ,"png"
    "showPlot": True,
    "saveFolder": "data/eval/tracking/plots/reprojectionErrorBarPlot",
    "saveName": "reprojectionErrorBarPlot",
    "methodsToEvaluate": ["cpd", "spr", "kpr"],  # "cpd", "spr", "kpr", "krcpd"
}
styleOpt = {
    "plotAspectRatio": "default",  # default , golden
    "legende": True,
    "legendPosition": "upper left",
    "colorPalette": thesisColorPalettes["viridis"],
    "meanMarkerSize": 5,
    "errorMarkerSize": 12,
    "errorMarkerAlpha": 0.2,
    "roundToDecimals": True,
    "grid": True,
}

# figure font configuration
latexFontSize_in_pt = 16
tex_fonts = {
    #    "pgf.texsystem": "pdflatex",
    # Use LaTeX to write all text
    "text.usetex": True,
    "font.family": "serif",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": latexFontSize_in_pt,
    "font.size": latexFontSize_in_pt,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": latexFontSize_in_pt,
    "xtick.labelsize": latexFontSize_in_pt,
    "ytick.labelsize": latexFontSize_in_pt,
}
plt.rcParams.update(tex_fonts)

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
    spacingXTicks=5,
    spacingErrorBars=3,
    capsize=3,
    legende=True,
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

    # generate plot
    if styleOpt["plotAspectRatio"] == "default":
        fig = plt.figure()
        ax = fig.add_subplot()
    elif styleOpt["plotAspectRatio"] == "golden":
        fig, ax = setupLatexPlot2D()
    else:
        raise NotImplementedError

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
    reprojectionErrors = []
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
        reprojectionErrorsPerMethod = []
        for index in sampleIdx:
            reprojectionErrorsPerMethod.append(
                reprojectionErrorEvaluationResults[method]["reprojectionErrors"][index]
            )
        reprojectionErrors.append(reprojectionErrorsPerMethod)
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
        methodColor = styleOpt["colorPalette"].to_rgba(
            i / (len(methodsToEvaluate) - 1)
        )[:3]
        ax.errorbar(
            positions[i],
            means[i],
            yerr=stds[i],
            fmt="o",
            markersize=styleOpt["meanMarkerSize"],
            label=method,
            capsize=capsize,
            color=methodColor,
        )
        for reprojectionErrorsPerFrame, correspondingPosition in zip(
            reprojectionErrors[i], positions[i]
        ):
            correspondingPositions = (
                np.ones(len(reprojectionErrorsPerFrame)) * correspondingPosition
            )
            plt.scatter(
                correspondingPositions,
                reprojectionErrorsPerFrame,
                s=styleOpt["errorMarkerSize"],
                marker="o",
                alpha=styleOpt["errorMarkerAlpha"],
                color=methodColor,
            )

    # Set the title, labels, and a legend
    plt.xlabel("frames")
    plt.ylabel("mean reprojection error in px")
    if styleOpt["roundToDecimals"]:
        framesToPlot = list(map(lambda x: (x // 10) * 10, framesToPlot))
    plt.xticks(XTicks, framesToPlot)
    # grid
    if styleOpt["grid"]:
        plt.grid(True)
    else:
        plt.grid(False)
    if legende:
        # create legend
        legendSymbols = []
        legendLabels = []
        for i, method in enumerate(methodsToEvaluate):
            symbolColor = styleOpt["colorPalette"].to_rgba(
                i / (len(methodsToEvaluate) - 1)
            )[:3]
            # error bar symbol
            errorBarHandle = configureLegendSymbol(
                "errorbar",
                markersize=5,
                color=symbolColor,
            )
            legendSymbols.append(errorBarHandle)
            legendLabels.append(method.upper())

        # # error bar symbol SPR
        # errorBarHandle_SPR = configureLegendSymbol(
        #     "errorbar",
        #     markersize=5,
        #     color=symbolColor,
        # )
        # legendSymbols.append(errorBarHandle_SPR)
        # legendLabels.append("SPR")

        # # error bar symbol KPR
        # errorBarHandle_KPR = configureLegendSymbol(
        #     "errorbar",
        #     markersize=5,
        #     color=symbolColor,
        # )
        # legendSymbols.append(errorBarHandle_KPR)
        # legendLabels.append("KPR")
        ax.legend(
            handles=legendSymbols,
            labels=legendLabels,
            loc=styleOpt["legendPosition"],
        )

    # Adjust layout to prevent clipping
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
        if controlOpt["saveAs"] == "png":
            # save as png
            plt.savefig(savePath)
        if controlOpt["saveAs"] == "pdf":
            plt.savefig(savePath + ".pdf", bbox_inches="tight")
        if controlOpt["saveAs"] == "tikz":
            # save as tixfigure
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
    for i, dataSetResult in enumerate(dataSetResults):
        createReprojectionErrorBoxPlots(
            dataSetResult,
            methodsToEvaluate=controlOpt["methodsToEvaluate"],
            legende=styleOpt["legende"],
        )
