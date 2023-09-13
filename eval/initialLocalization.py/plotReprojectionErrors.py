import sys
import os
import matplotlib.pyplot as plt
import numpy as np


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
    "resultsToLoad": 0,
    "save": True,
    "showPlot": True,
    "block": False,
    "saveFolder": "data/eval/initialLocalization/plots/initialLocalizationResults",
    "saveName": "initialLocalizationResultImg",
    "verbose": True,
}

resultFolderPaths = [
    "data/eval/initialLocalization/results/20230516_112207_YShape",
    "data/eval/initialLocalization/results/20230807_150735_partial",
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
        if controlOpt["showPlot"]:
            fig, ax = createReprojectionErrorBoxPlot(
                frames=frames,
                means=means,
                stds=stds,
            )
        plt.show()
        # save plot
        if controlOpt["save"]:
            dataSetPath = result["dataSetPath"]
            id = "_".join(resultFile.split("_")[0:3])
            fileName = id + "_" + controlOpt["saveName"]
            dataSetName = dataSetPath.split("/")[-2]
            folderPath = os.path.join(controlOpt["saveFolder"], dataSetName)
            savePath = os.path.join(folderPath, fileName)
            if not os.path.exists(folderPath):
                os.makedirs(folderPath, exist_ok=True)
            eval.saveImage(rgbImg, savePath)
        if controlOpt["verbose"]:
            print(
                "Saved image for initial localization result {} for at {}.".format(
                    resultFile, savePath
                )
            )
