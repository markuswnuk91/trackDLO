import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from collections import defaultdict
from scipy.optimize import curve_fit
from scipy.stats import norm
import tikzplotlib

from matplotlib.ticker import MaxNLocator

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
    "makeScatterPlot": True,
    "makeHistogramPlot": True,
    "save": True,
    "saveAsTikz": False,  # does not work with dashed lines, and does not plot legend
    "saveAsPGF": True,
    "verbose": True,
}

saveOpt = {
    "saveFolder": "data/eval/graspingAccuracy/plots/graspingAccuracyEvaluationResults",
    "saveFileNameScatterPlot": "graspingErrorsScatterPlot",
    "saveFileNameHistogramTranslational": "translationalGraspingErrorsHistogram",
    "saveFileNameHistogramRotaional": "rotationalGraspingErrorsHistogram",
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
    "histogramFitLineStyle": "--",
}
textwidth_in_pt = 483.6969
figureScaling = 0.45
latexFontSize_in_pt = 14
latexFootNoteFontSize_in_pt = 10
desiredFigureWidth = figureScaling * textwidth_in_pt
desiredFigureHeight = figureScaling * textwidth_in_pt
tex_fonts = {
    #    "pgf.texsystem": "pdflatex",
    # Use LaTeX to write all text
    "text.usetex": True,
    "font.family": "serif",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": latexFontSize_in_pt,
    "font.size": latexFontSize_in_pt,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": latexFootNoteFontSize_in_pt,
    "xtick.labelsize": latexFootNoteFontSize_in_pt,
    "ytick.labelsize": latexFootNoteFontSize_in_pt,
}
if controlOpt["saveAsPGF"]:
    matplotlib.use("pgf")
    matplotlib.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "font.family": "serif",
            "text.usetex": True,
            "pgf.rcfonts": False,
        }
    )
    matplotlib.rcParams.update(tex_fonts)
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


def tikzplotlib_fix_ncols(obj):
    """
    workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
    """
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)


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
    fig, ax = setupLatexPlot2D(
        figureWidth=1.5 * desiredFigureWidth, figureHeight=desiredFigureHeight
    )
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

    # axis legend
    ax.set_xlabel(r"translational errors in $m$")
    if controlOpt["saveAsPGF"]:
        ax.set_ylabel(r"rotational error in $^\circ$")
    else:
        ax.set_ylabel(r"rotational error in $°$")
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
    errors,
    correspondingMethods,
    correspondingModelNames,
    n_bins=20,
    mode="translational",
    plotLegend=True,
):
    # Ensure that the two lists have the same length
    assert len(errors) == len(correspondingMethods)

    # Create a defaultdict with lists as default values
    grouped_vals = defaultdict(list)

    # Iterate through both lists simultaneously using zip
    for method, val in zip(correspondingMethods, errors):
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
    fig, ax = setupLatexPlot2D(
        figureWidth=desiredFigureWidth, figureHeight=desiredFigureHeight
    )
    hist_handle = ax.hist(
        x, n_bins, density=False, histtype="bar", color=histogramColors
    )

    # # plot gaussian

    x_axis = np.linspace(hist_handle[1][0], hist_handle[1][-1], 1000)
    fitLineStyle = styleOpt["histogramFitLineStyle"]
    scalefactor = len(x) * np.mean(np.diff(hist_handle[1]))
    for method in grouped_vals:
        mu = np.mean(grouped_vals[method])
        std = np.std(grouped_vals[method])
        p = norm.pdf(x_axis, mu, std)
        # scale density to counts
        ax.plot(
            x_axis,
            p * scalefactor,
            color=styleOpt["methodColors"][method],
            linestyle=fitLineStyle,
        )

    # add density as secondary axis
    def counts_to_density(counts):
        return counts / scalefactor

    def density_to_counts(density):
        return density * scalefactor

    secax_y = ax.secondary_yaxis(
        "right", functions=(counts_to_density, density_to_counts)
    )

    # set axis labels
    if mode == "translational":
        ax.set_xlabel(r"translational errors in $m$")
    elif mode == "rotational":
        if controlOpt["saveAsPGF"]:
            ax.set_xlabel(r"rotational errors in $^\circ$")
        else:
            ax.set_xlabel(r"rotational errors in $^°$")
    ax.set_ylabel("counts")
    secax_y.set_ylabel(r"probability density")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))  # make y axis only integers
    if plotLegend:
        # create legend
        legendSymbols = []
        for method in grouped_vals:
            legendSymbol = Patch(
                facecolor=styleOpt["methodColors"][method],
                label=method,
            )
            legendSymbols.append(legendSymbol)
        for method in grouped_vals:
            legendSymbol = Line2D(
                [],
                [],
                color=styleOpt["methodColors"][method],
                linestyle=fitLineStyle,
                label="fitted gaussian, " + method,
            )
            legendSymbols.append(legendSymbol)
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
        if controlOpt["save"]:
            saveFileNameScatter = saveOpt["saveFileNameScatterPlot"]
            saveFolder = saveOpt["saveFolder"]
            savePathScatter = os.path.join(saveFolder, saveFileNameScatter)
            if not os.path.exists(saveFolder):
                os.makedirs(saveFolder, exist_ok=True)
            fig_scatterPlot.savefig(savePathScatter)
            # save as tixfigure
            if controlOpt["saveAsTikz"]:
                tikzplotlib_fix_ncols(fig_scatterPlot)
                tikzplotlib.save(
                    figure=fig_scatterPlot,
                    filepath=savePathScatter + ".tex",
                )
            if controlOpt["saveAsPGF"]:
                fig_scatterPlot.savefig(
                    savePathScatter + ".pgf",
                    bbox_inches="tight",
                )
        # if controlOpt["showPlot"]:
        #     plt.show(block=True)
    if controlOpt["makeHistogramPlot"]:
        fig_histogram_trans, ax_histogram_trans = graspingErrorsHistogram(
            errors=translationalGraspingErrors,
            correspondingMethods=methods,
            correspondingModelNames=models,
            mode="translational",
        )
        fig_histogram_rot, ax_histogram_rot = graspingErrorsHistogram(
            errors=rotationalGraspingErrors,
            correspondingMethods=methods,
            correspondingModelNames=models,
            mode="rotational",
            plotLegend=False,
        )
        if controlOpt["save"]:
            saveFileNameTrans = saveOpt["saveFileNameHistogramTranslational"]
            saveFileNameRot = saveOpt["saveFileNameHistogramRotaional"]
            saveFolder = saveOpt["saveFolder"]
            savePathTrans = os.path.join(saveFolder, saveFileNameTrans)
            savePathRot = os.path.join(saveFolder, saveFileNameRot)
            if not os.path.exists(saveFolder):
                os.makedirs(saveFolder, exist_ok=True)
            fig_histogram_trans.savefig(savePathTrans)
            fig_histogram_rot.savefig(savePathRot)
            if controlOpt["saveAsPGF"]:
                fig_histogram_trans.savefig(savePathTrans + ".pgf", bbox_inches="tight")
                fig_histogram_rot.savefig(savePathRot + ".pgf", bbox_inches="tight")
            # save as tixfigure
            if controlOpt["saveAsTikz"]:
                tikzplotlib_fix_ncols(fig_histogram_trans)
                tikzplotlib.save(
                    figure=fig_histogram_trans, filepath=savePathTrans + ".tex"
                )
                tikzplotlib_fix_ncols(fig_histogram_rot)
                tikzplotlib.save(
                    figure=fig_histogram_rot, filepath=savePathRot + ".tex"
                )
    if controlOpt["showPlot"]:
        plt.show(block=True)
    if controlOpt["verbose"]:
        print("Finished result generation.")
