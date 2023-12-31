import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib
import traceback

try:
    sys.path.append(os.getcwd().replace("/eval", ""))
    from src.evaluation.initialLocalization.initialLocalizationEvaluation import (
        InitialLocalizationEvaluation,
    )
    from src.visualization.plot3D import *
    from src.visualization.colors import *
except:
    print("Imports for plotting localization results 2D failed.")
    raise

global eval
eval = InitialLocalizationEvaluation()

controlOpt = {
    "dataSetsToLoad": [0],
    "resultsToLoad": [1],
    "save": True,
    "showPlot": True,
    "block": True,
    "saveFolder": "data/eval/initialLocalization/plots/initialLocalizationResults3D",
    "saveName": "initialLocalizationResult",
    "saveAsTikz": True,
    "saveAsPGF": False,
    "verbose": True,
    "visualizeInputPointCloud": True,
}

resultFolderPaths = [
    "data/eval/initialLocalization/results/20230603_143937_modelY",
    "data/eval/initialLocalization/results/20230516_112207_YShape",
    "data/eval/initialLocalization/results/20230807_150735_partial",
    "data/eval/initialLocalization/results/20230516_113957_Partial",
    "data/eval/initialLocalization/results/20230516_115857_arena",
    "data/eval/initialLocalization/results/20230603_140143_arena",
]

styleOpt = {
    "lineColor": thesisColors["blue"],
    "lineThickness": 8,
    "circleColor": thesisColors["blue"],
    "circleRadius": 20,
    "pointCloudColor": thesisColors["red"],
    "pointCloudAlpha": 0.3,
    "pointCloudSize": 1,
    "pointCloudMarkerStyle": ".",
    "pointCloudEdgeColor": thesisColors["red"],
    "azimuth": 70,
    "elevation": 30,
}

saveOpt = {
    "dpi": 100,
    "bbox_inches": "tight",
    "pad_inches": 0,
}


def makePlot(result):
    fig, ax = setupLatexPlot3D()
    if controlOpt["visualizeInputPointCloud"]:
        eval.config = result["config"]
        Y, _ = eval.getPointCloud(result["frame"], result["dataSetPath"])
        plotPointSet(
            ax=ax,
            X=Y,
            size=styleOpt["pointCloudSize"],
            color=styleOpt["pointCloudColor"],
            alpha=styleOpt["pointCloudAlpha"],
            markerStyle=styleOpt["pointCloudMarkerStyle"],
            edgeColor=styleOpt["pointCloudEdgeColor"],
        )
    ax = eval.plotLocalizationResult3D(ax=ax, result=result)
    if styleOpt["elevation"] and styleOpt["azimuth"] is not None:
        ax.view_init(elev=styleOpt["elevation"], azim=styleOpt["azimuth"])
    return fig, ax


if __name__ == "__main__":
    if controlOpt["dataSetsToLoad"][0] == -1:
        dataSetsToEvaluate = resultFolderPaths
    else:
        dataSetsToEvaluate = [
            dataSetPath
            for i, dataSetPath in enumerate(resultFolderPaths)
            if i in controlOpt["dataSetsToLoad"]
        ]
    # load results
    for resultFolderPath in dataSetsToEvaluate:
        if controlOpt["resultsToLoad"][0] == -1:
            resultFiles = eval.list_result_files(resultFolderPath)
        else:
            resultFiles = eval.list_result_files(resultFolderPath)
            resultFiles = [
                file
                for i, file in enumerate(resultFiles)
                if i in controlOpt["resultsToLoad"]
            ]
        failedFrames = []
        for resultFile in resultFiles:
            resultFilePath = os.path.join(resultFolderPath, resultFile)
            result = eval.loadResults(resultFilePath)
            try:
                # make plot
                lineColor = styleOpt["lineColor"]
                circleColor = styleOpt["circleColor"]
                lineThickness = styleOpt["lineThickness"]
                circleRadius = styleOpt["circleRadius"]
                fig, ax = makePlot(result)
                # save image
                if controlOpt["save"]:
                    id = "_".join(resultFile.split("_")[0:3])
                    fileName = id + "_" + controlOpt["saveName"]
                    dataSetName = result["dataSetPath"].split("/")[-2]
                    folderPath = os.path.join(controlOpt["saveFolder"], dataSetName)
                    savePath = os.path.join(folderPath, fileName)
                    if not os.path.exists(folderPath):
                        os.makedirs(folderPath, exist_ok=True)
                    plt.savefig(
                        savePath,
                        dpi=saveOpt["dpi"],
                        bbox_inches=saveOpt["bbox_inches"],
                        pad_inches=saveOpt["pad_inches"],
                    )
                    if controlOpt["saveAsTikz"]:
                        tikzplotlib.save(savePath + ".tex")
                    if controlOpt["saveAsPGF"]:
                        raise NotImplementedError
                        # plt.savefig(filePath, format="pgf", bbox_inches="tight", pad_inches=0)
                    if controlOpt["verbose"]:
                        print(
                            "Saved image for initial localization result {} for at {}.".format(
                                resultFile, savePath
                            )
                        )
                if controlOpt["showPlot"]:
                    plt.show(block=controlOpt["block"])
                plt.close("all")
            except:
                failedFrames.append(result["frame"])
                traceback.print_exc()
        if len(failedFrames) > 0:
            print("Failed on frames {}".format(failedFrames))
