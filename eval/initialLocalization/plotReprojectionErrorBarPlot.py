import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib

try:
    sys.path.append(os.getcwd().replace("/eval", ""))
    from src.evaluation.initialLocalization.initialLocalizationEvaluation import (
        InitialLocalizationEvaluation,
    )
    from src.visualization.colors import *
    from src.visualization.plotUtils import *
except:
    print("Imports for plotting localization results 2D failed.")
    raise

global eval
global REPROJECTION_ERROR_THESHOLD_MEAN
global REPROJECTION_ERROR_THESHOLD_STD

eval = InitialLocalizationEvaluation()
REPROJECTION_ERROR_THESHOLD_MEAN = 100
REPROJECTION_ERROR_THESHOLD_STD = 70

controlOpt = {
    "resultsToLoad": [-1],
    "save": True,
    "showPlot": True,
    "block": False,
    "saveFolder": "data/eval/initialLocalization/plots/reprojectionErrorBarPlots",
    "saveName": "reprojectionErrors",
    "saveAs": "PDF",  # "TIKZ", "PGF", "PNG", "PDF"
    "verbose": True,
}

resultFolderPaths = [
    "data/eval/initialLocalization/results/20230516_112207_YShape",
    "data/eval/initialLocalization/results/20230516_113957_Partial",
    "data/eval/initialLocalization/results/20230516_115857_arena",
    "data/eval/initialLocalization/results/20230603_143937_modelY",
    "data/eval/initialLocalization/results/20230807_150735_partial",
    "data/eval/initialLocalization/results/20230603_140143_arena",
]

styleOpt = {
    "plotLegendInFrame": 0,
    "colorPalette": thesisColorPalettes["viridis"],
    "outlierColor": [0.6, 0.6, 0.6],
    "xAxisRange": [-3, 52],
    "yAxisRange": [-50, 600],
    "meanMarkerSize": 5,
    "errorMarkerSize": 12,
    "alpha": 0.2,
    "plotWidth": 1.5 * 0.6 * 6.69423,
    "plotHeight": 0.6 * 6.69423,
    "fontsize": 16,
}


def setPlotOptions():
    if controlOpt["saveAs"] == "PGF":
        matplotlib.rcParams.update(
            {
                "pgf.texsystem": "pdflatex",
                "font.family": "serif",
                "text.usetex": True,
                "pgf.rcfonts": False,
            }
        )
    elif controlOpt["saveAs"] == "PDF":
        # plt.rcParams["text.latex.preamble"] = [r"\usepackage{DejaVuSansMono}"]
        # Options
        params = {
            "text.usetex": True,
            "font.size": styleOpt["fontsize"],
            "font.family": "serif",
            # "text.latex.unicode": True,
        }
        plt.rcParams.update(params)
    return


def createReprojectionErrorBoxPlot(
    frames,
    means,
    stds,
    reprojectionErrors,
    capsize=3,
    createLegend=False,
):
    fig, ax = plt.subplots(figsize=(styleOpt["plotWidth"], styleOpt["plotHeight"]))

    for i, (frame, mean, std) in enumerate(zip(frames, means, stds)):
        frameColor = styleOpt["colorPalette"].to_rgba(i / len(frames))[:3]
        if (
            mean < REPROJECTION_ERROR_THESHOLD_MEAN
            and std < REPROJECTION_ERROR_THESHOLD_STD
        ):
            meanColor = frameColor
        else:
            meanColor = styleOpt["outlierColor"]
        ax.errorbar(
            frame,
            mean,
            yerr=std,
            fmt="o",
            markersize=styleOpt["meanMarkerSize"],
            capsize=capsize,
            color=meanColor,
            label="errorBar_" + str(i),
            # mfc=color,
            # mec=color,
            # ms=20,
            # mew=4
        )
        for error in reprojectionErrors[i]["reprojectionErrors"]:
            plt.scatter(
                frame,
                error,
                s=styleOpt["errorMarkerSize"],
                marker="o",
                alpha=styleOpt["alpha"],
                color=meanColor,
            )

    # Set the title, labels, grind, and legend
    plt.grid(True)
    ax.set_xlim(styleOpt["xAxisRange"])
    ax.set_ylim(styleOpt["yAxisRange"])
    plt.xlabel("frame")
    plt.ylabel("reprojection error in px")
    # if styleOpt["legende"]:
    #     plt.legend()
    plt.tight_layout()
    if createLegend:
        # create legend
        symbolColor = styleOpt["colorPalette"].to_rgba(0)[:3]
        legendSymbols = []
        legendLabels = []
        # error bar symbol
        errorBarHandle = configureLegendSymbol(
            "errorbar",
            markersize=5,
            color=symbolColor,
        )
        legendSymbols.append(errorBarHandle)
        legendLabels.append("mean + std")

        # outlier symbol
        errorBarSymbol = configureLegendSymbol(
            "errorbar", markersize=5, color=styleOpt["outlierColor"]
        )
        legendSymbols.append(errorBarSymbol)
        legendLabels.append("outlier")

        # point symbol
        errorPointHandle = configureLegendSymbol_Point(
            markersize=4, color=symbolColor, alpha=styleOpt["alpha"]
        )
        legendSymbols.append(errorPointHandle)
        legendLabels.append("reprojection error")

        ax.legend(
            handles=legendSymbols,
            labels=legendLabels,
            # loc="upper right",
            loc="upper left",
        )
    return fig, ax


if __name__ == "__main__":
    setPlotOptions()
    if controlOpt["resultsToLoad"][0] == -1:
        dataSetsToEvaluate = resultFolderPaths
    else:
        dataSetsToEvaluate = [
            dataSetPath
            for i, dataSetPath in enumerate(resultFolderPaths)
            if i in controlOpt["resultsToLoad"]
        ]
    # load results
    for i, resultFolderPath in enumerate(resultFolderPaths):
        resultFiles = eval.list_result_files(resultFolderPath)
        reprojectionErrorResults = []
        frames = []
        means = []
        stds = []
        for resultFile in resultFiles:
            resultFilePath = os.path.join(resultFolderPath, resultFile)
            result = eval.loadResults(resultFilePath)
            reprojectionErrorResult = eval.calculateReprojectionError(result)
            reprojectionErrorResults.append(reprojectionErrorResult)
            frames.append(result["frame"])
            means.append(reprojectionErrorResult["meanReprojectionError"])
            stds.append(reprojectionErrorResult["stdReprojectionError"])

        frames = np.array(frames)
        means = np.array(means)
        stds = np.array(stds)
        if i == styleOpt["plotLegendInFrame"]:
            fig, ax = createReprojectionErrorBoxPlot(
                frames=frames,
                means=means,
                stds=stds,
                reprojectionErrors=reprojectionErrorResults,
                createLegend=True,
            )
        else:
            fig, ax = createReprojectionErrorBoxPlot(
                frames=frames,
                means=means,
                stds=stds,
                reprojectionErrors=reprojectionErrorResults,
                createLegend=False,
            )
        # # save plot
        dataSetPath = result["dataSetPath"]
        dataSetName = dataSetPath.split("/")[-2]
        folderPath = os.path.join(controlOpt["saveFolder"], dataSetName)
        if not os.path.exists(folderPath):
            os.makedirs(folderPath, exist_ok=True)
        fileName = controlOpt["saveName"]
        savePath = os.path.join(folderPath, fileName)
        if controlOpt["save"]:
            if controlOpt["saveAs"] == "PNG":
                plt.savefig(savePath)
            if controlOpt["saveAs"] == "TIKZ":
                tikzplotlib.save(savePath + ".tex")
            if controlOpt["saveAs"] == "PGF":
                plt.savefig(savePath + ".pgf")
            if controlOpt["saveAs"] == "PGF":
                plt.savefig(savePath + ".pgf")
            if controlOpt["saveAs"] == "PDF":
                plt.savefig(savePath + ".pdf")
            if controlOpt["verbose"]:
                print(
                    "Saved image for initial localization result {} for at {}.".format(
                        resultFile, savePath
                    )
                )
        if controlOpt["showPlot"]:
            plt.show(block=controlOpt["block"])
    if controlOpt["showPlot"]:
        plt.show(block=True)
    print("Done.")
