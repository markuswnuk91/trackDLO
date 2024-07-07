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
    "save": True,
    "saveAsTikz": False,  # does not work with dashed lines, and does not plot legend
    "saveAsPGF": False,
    "verbose": True,
}

saveOpt = {
    "saveFolder": "data/eval/graspingAccuracy/plots/graspingAccuracyEvaluationResults",
    "saveFileName": "graspingAccuracyBoxPlot",
}

styleOpt = {
    "methodColors": {
        "cpd": thesisColors["susieluMagenta"],
        "spr": thesisColors["susieluGold"],
        "kpr": thesisColors["susieluBlue"],
    },
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


def plotGraspingAccuracyBoxPlot(
    graspingErrors, correspondingMethods, correspondingModelNames, medianColor=[0, 0, 0]
):
    methodsToEvaluate = list(set(correspondingMethods))
    modelNames = list(set(correspondingModelNames))

    # Ensure that the two lists have the same length
    assert len(graspingErrors) == len(correspondingMethods)

    # Create a defaultdict with lists as default values
    grouped_vals = defaultdict(list)

    # Iterate through both lists simultaneously using zip
    for method, val in zip(correspondingMethods, graspingErrors):
        grouped_vals[method].append(val)

    # Convert defaultdict to a regular dict (optional)
    grouped_vals = dict(grouped_vals)

    fig, ax = setupLatexPlot2D()

    data = []
    labels = []
    colors = []

    facealpha = 0.3
    linealpha = 1

    for method in grouped_vals:
        data.append(grouped_vals[method])
        labels.append(method.upper())
        colors.append(styleOpt["methodColors"][method])
    boxPlot = ax.boxplot(
        data,
        patch_artist=True,  # fill with color
        labels=labels,
    )
    for median, color in zip(boxPlot["medians"], colors):
        median.set_color(medianColor)
    for patch, color in zip(boxPlot["boxes"], colors):
        patch.set_facecolor(np.concatenate((color, [facealpha])))
        patch.set_edgecolor(np.concatenate((color, [linealpha])))
    for color in colors:
        for item in ["whiskers", "fliers", "caps"]:
            plt.setp(item[item], color)
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
                graspingAccuracyError = eval.evaluateGraspingAccuracy(
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
    plotGraspingAccuracyBoxPlot(
        graspingErrors=translationalGraspingErrors,
        correspondingMethods=methods,
        correspondingModelNames=models,
    )
    if controlOpt["showPlot"] and not controlOpt["saveAsPGF"]:
        plt.show(block=True)
    if controlOpt["verbose"]:
        print("Finished plotting.")
