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
except:
    print("Imports for plotting localization results 2D failed.")
    raise

global eval
eval = InitialLocalizationEvaluation()

controlOpt = {
    "resultsToLoad": [-1],
    "save": True,
    "showPlot": True,
    "block": False,
    "saveFolder": "data/eval/initialLocalization/plots/reprojectionErrorBarPlots",
    "saveName": "initialLocalizationResultImg",
    "saveAsTikz": True,
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

styleOpt = {}


def createReprojectionErrorBoxPlot(
    frames,
    means,
    stds,
    capsize=3,
):
    fig, ax = plt.subplots()
    ax.errorbar(
        frames,
        means,
        yerr=stds,
        fmt="o",
        capsize=capsize,
    )

    # Set the title, labels, and a legend
    plt.xlabel("frames")
    plt.ylabel("mean reprojection error in px")
    # if styleOpt["legende"]:
    #     plt.legend()
    plt.tight_layout()

    return fig, ax


if __name__ == "__main__":
    if controlOpt["resultsToLoad"][0] == -1:
        dataSetsToEvaluate = resultFolderPaths
    else:
        dataSetsToEvaluate = [
            dataSetPath
            for i, dataSetPath in enumerate(resultFolderPaths)
            if i in controlOpt["resultsToLoad"]
        ]
    # load results
    for resultFolderPath in resultFolderPaths:
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
        fig, ax = createReprojectionErrorBoxPlot(
            frames=frames,
            means=means,
            stds=stds,
        )

        # # save plot
        if controlOpt["save"]:
            dataSetPath = result["dataSetPath"]
            id = "_".join(resultFile.split("_")[0:3])
            fileName = id + "_" + controlOpt["saveName"]
            dataSetName = dataSetPath.split("/")[-2]
            folderPath = os.path.join(controlOpt["saveFolder"], dataSetName)
            savePath = os.path.join(folderPath, fileName)
            if not os.path.exists(folderPath):
                os.makedirs(folderPath, exist_ok=True)
            plt.savefig(savePath)
            # save as tixfigure
            if controlOpt["saveAsTikz"]:
                tikzplotlib.save(savePath + ".tex")
        if controlOpt["verbose"]:
            print(
                "Saved image for initial localization result {} for at {}.".format(
                    resultFile, savePath
                )
            )
        if controlOpt["showPlot"]:
            plt.show()
