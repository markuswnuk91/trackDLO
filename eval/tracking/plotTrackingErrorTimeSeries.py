import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib
import pickle
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FormatStrFormatter
import matplotlib.cm as cm

try:
    sys.path.append(os.getcwd().replace("/eval", ""))
    from src.evaluation.tracking.trackingEvaluation import TrackingEvaluation
    from src.visualization.plot2D import *
    from src.visualization.plotUtils import *
    from src.visualization.colors import *
except:
    print("Imports for plotting script tracking error time series failed.")
    raise

global eval
eval = TrackingEvaluation()

controlOpt = {
    "resultsToLoad": [1],  # [0,1,2]
    "save": True,
    "saveAs": "pdf",  # tikz, pdf, png
    "showPlot": True,
    "saveFolder": "data/eval/tracking/plots/trackingErrorTimeSeries",
    "saveName": "trackingErrorTimeSeries",
    "methodsToEvaluate": ["cpd", "spr", "kpr"],  # "cpd", "spr", "kpr", "krcpd"
}
styleOpt = {
    "plotAspectRatio": "default",  # default , golden
    "legende": True,
    "legendPosition": "upper left",
    "colorPalette": thesisColorPalettes["viridis"],
    "lineStyles": ["-", "-", "-"],  # line styles for CPD, SPR, KPR respectively
    "movingAverageWindowSize": 10,
    "movingAverageAlpha": 0.9,
    "movingAverageLineWidth": 1,
    "trackingErrorsAlpha": 0.3,
    "trackingErrorsLineStyle": "-",
    "trackingErrorsLineWidth": 3,
    "highlightFrames": [
        [400, 560, 630],
        [120, 145, 280],
        [55, 390, 465],
    ],  # list of list, each sublist for one data set
    "highlightLabels": [
        ["$t_1$", "$t_2$", "$t_3$"],
        ["$t_1$", "$t_2$", "$t_3$"],
        ["$t_1$", "$t_2$", "$t_3$"],
    ],  # list of list, each sublist for one data set
    "highlightColor": [1, 0, 0],
    "highlightTextColor": [0, 0, 0],
    "highlightAlpha": 0.5,
    "highlightLineStyles": [["--", "--", "--"], ["--", "--", "--"], ["--", "--", "--"]],
    "grid": True,
    "yAxisDescription": "tracking error in cm",
    "unitScalingFactor": 100,  # scale tracking errors to cm
    "cutOffThreshold": 5,  # leaves out last n values when plotting to avoid unreasonable dropoff from moving average
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


def createTrackingErrorTimeSeriesPlot(
    dataSetResult,
    methodsToEvaluate=None,
    lineStyles=None,
    colorPalette=None,
    highlightFrames=None,
    highlightColor=[1, 0, 0],
    highlightAlpha=1.0,
    highlightLineStyles=None,
    highlightLabels=None,
    highlightLabelFontSize=None,
    highlightTextColor=None,
    grid=None,
):
    """
    Creates a tracking error plot over all methods, with optional customization for highlight lines
    (e.g., frames to highlight with vertical lines).

    Parameters:
    - dataSetResult: Dictionary containing tracking results and data set path.
    - methodsToEvaluate: List of methods to plot (default is all methods).
    - lineColors: Custom colors for the methods' lines (optional).
    - highlightFrames: List of frames to highlight with vertical lines.
    - highlightColor: Color of the highlight vertical lines (default is red).
    - highlightAlpha: Alpha (transparency) of the highlight vertical lines (default is 1.0).
    - highlightLineStyles: List of line styles for the highlights (optional, default is solid).
    - highlightLabels: List of labels for each highlight (optional).
    """

    trackingResults = dataSetResult["trackingResults"]
    colorPalette = plt.cm.ScalarMappable(
        cmap=matplotlib.colormaps["viridis"],
        norm=matplotlib.colors.Normalize(vmin=0, vmax=1),
    )
    methodsToEvaluate = (
        list(trackingResults.keys()) if methodsToEvaluate is None else methodsToEvaluate
    )
    lineStyles = ["-", "-", "-"] if lineStyles is None else lineStyles
    highlightFrames = [] if highlightFrames is None else highlightFrames
    highlightLineStyles = (
        ["-"] * len(highlightFrames)
        if highlightLineStyles is None
        else highlightLineStyles
    )
    highlightLabels = (
        [""] * len(highlightFrames) if highlightLabels is None else highlightLabels
    )
    highlightTextColor = [0, 0, 0] if highlightTextColor is None else highlightTextColor
    highlightLabelFontSize = (
        10 if highlightLabelFontSize is None else highlightLabelFontSize
    )
    grid = True if grid is None else grid

    if styleOpt["plotAspectRatio"] == "default":
        fig = plt.figure()
        ax = fig.add_subplot()
    elif styleOpt["plotAspectRatio"] == "golden":
        fig, ax = setupLatexPlot2D()
    else:
        raise NotImplementedError

    trackingErrorLines = []
    ymax = 0  # Initialize a variable to track the maximum y value

    for idx, method in enumerate(methodsToEvaluate):
        trackingErrors = (
            np.array(eval.calculateTrackingErrors(trackingResults[method]))
            * styleOpt["unitScalingFactor"]
        )
        trackingErrorsMovingAvg = np.convolve(
            trackingErrors,
            np.ones(styleOpt["movingAverageWindowSize"])
            / styleOpt["movingAverageWindowSize"],
            mode="same",
        )
        color = colorPalette.to_rgba(idx / (len(methodsToEvaluate) - 1))[:3]
        (trackingErrorLine,) = ax.plot(
            list(range(len(trackingErrors[: -styleOpt["cutOffThreshold"]]))),
            trackingErrors[: -styleOpt["cutOffThreshold"]],
            color=color,
            linestyle=styleOpt["trackingErrorsLineStyle"],
            linewidth=styleOpt["trackingErrorsLineWidth"],
            alpha=styleOpt["trackingErrorsAlpha"],
        )
        (trackingErrorLine,) = ax.plot(
            list(range(len(trackingErrors[: -styleOpt["cutOffThreshold"]]))),
            trackingErrorsMovingAvg[: -styleOpt["cutOffThreshold"]],
            color=color,
            linestyle=lineStyles[idx],
            linewidth=styleOpt["movingAverageLineWidth"],
            alpha=styleOpt["movingAverageAlpha"],
        )
        trackingErrorLine.set_label(method.upper())
        trackingErrorLines.append(trackingErrorLine)
        ymax = max(ymax, max(trackingErrors))  # Update ymax

    # for highlightFrame in highlightFrames:
    #     ax.axvline(x=highlightFrame, color=highlightColor)

    # Plot vertical lines with labels above them
    for i, highlightFrame in enumerate(highlightFrames):
        linestyle = highlightLineStyles[i] if i < len(highlightLineStyles) else "-"
        label = highlightLabels[i] if i < len(highlightLabels) else ""

        # Plot vertical line
        ax.axvline(
            x=highlightFrame,
            color=highlightColor,
            alpha=highlightAlpha,
            linestyle=linestyle,
        )

        # Add label above the vertical line at a slightly higher y position
        if label:
            ax.text(
                highlightFrame,
                ymax * 1.05,
                label,
                rotation=0,
                color="black",
                ha="center",
                va="bottom",
                fontsize=highlightLabelFontSize,
            )

    # grid
    if grid:
        plt.grid(True)

    # create legend
    if styleOpt["legende"]:
        # make legend
        ax.legend(loc=styleOpt["legendPosition"])

    plt.xlabel("frames")
    plt.ylabel(styleOpt["yAxisDescription"])

    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    # Adjust layout to prevent clipping
    plt.tight_layout()

    # # Use scientific notation on the y-axis
    # formatter = ScalarFormatter(useMathText=True)
    # formatter.set_scientific(True)
    # formatter.set_powerlimits(
    #     (-3, 3)
    # )  # Adjust the limits for when the notation is triggered
    # ax.yaxis.set_major_formatter(formatter)

    # # Force display of the scale factor (e.g., "*10^-3") above the y-axis
    # ax.ticklabel_format(style="sci", axis="y", scilimits=(-3, 3))
    # Customize the offset text to use `\cdot` instead of `x`
    # plt.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    # Adjust offset text and its position for better alignment
    # ax.yaxis.get_offset_text().set_x(-0.05)
    # ax.yaxis.get_offset_text().set_y(1.1)
    # offset_text = ax.yaxis.get_offset_text()
    # offset_text.set_text("abs")

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
        if controlOpt["saveAs"] == "png":
            plt.savefig(savePath)
        # save as tixfigure
        if controlOpt["saveAs"] == "tikz":
            tikzplotlib.save(savePath + ".tex")
        if controlOpt["saveAs"] == "pdf":
            plt.savefig(savePath + ".pdf", bbox_inches="tight")
    if controlOpt["showPlot"]:
        plt.show(block=True)


if __name__ == "__main__":
    # load all results
    results = []
    for resultFilePath in [resultFolderPaths[x] for x in controlOpt["resultsToLoad"]]:
        resultFilePath = os.path.join(resultFilePath, resultFileName)
        result = loadResult(resultFilePath)
        results.append(result)
    # create plot
    for i, result in enumerate(results):
        createTrackingErrorTimeSeriesPlot(
            dataSetResult=result,
            methodsToEvaluate=controlOpt["methodsToEvaluate"],
            lineStyles=styleOpt["lineStyles"],
            highlightFrames=styleOpt["highlightFrames"][controlOpt["resultsToLoad"][i]],
            highlightColor=styleOpt["highlightColor"],
            highlightLabels=styleOpt["highlightLabels"][controlOpt["resultsToLoad"][i]],
            highlightAlpha=styleOpt["highlightAlpha"],
            highlightLineStyles=styleOpt["highlightLineStyles"][
                controlOpt["resultsToLoad"][i]
            ],
            highlightLabelFontSize=latexFontSize_in_pt,
            grid=styleOpt["grid"],
        )
    print("Done.")
