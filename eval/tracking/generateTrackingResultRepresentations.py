import sys
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib

try:
    sys.path.append(os.getcwd().replace("/eval", ""))
    from src.evaluation.tracking.trackingEvaluation import (
        TrackingEvaluation,
    )
except:
    print("Imports for Tracking Result Evaluation failed.")
    raise


# script control parameters
controlOptions = {"tabularizeResults": False, "createTrackingTimeSeriesPlots": True}
resultFileName = "result"
resultFolderPaths = [
    "data/eval/tracking/results/20230524_171237_ManipulationSequences_mountedWireHarness_modelY"
]
resultsToLoad = [0]
savePathsForRepresentations = {
    "trackingErrorTimeSeries": "data/eval/tracking/trackingTimeSeriesPlots"
}


def tabularizeResults(accumulatedResults, methodsToPrint=["cpd", "spr", "krcpd"]):
    trackingErrorScaleFactor = 100  # cm
    geometricErrorScaleFactor = 100  # cm
    runtimeScaleFactor = 1000  # mm
    for result in accumulatedResults:
        modelID = eval.getModelID(
            result["initializationResult"]["modelParameters"]["modelInfo"]["name"]
        )
        dataSetTableRowEntries = {
            "modelID": modelID,
            "n_tracked_frames": len(result["trackingResults"][0]["frames"]),
            "n_labeld_frames": len(
                result["trackingEvaluationResults"]["cpd"]["reprojectionErrors"][
                    "frames"
                ]
            ),
            "n_methods": len(methodsToPrint),
        }
        for trackingMethod in list(result["trackingEvaluationResults"].keys()):
            if trackingMethod not in methodsToPrint:
                continue
            dataSetTableRowEntries[trackingMethod] = {
                "trackingError": trackingErrorScaleFactor
                * np.mean(
                    result["trackingEvaluationResults"][trackingMethod][
                        "trackingErrors"
                    ]
                ),
                "geometricError": geometricErrorScaleFactor
                * np.mean(
                    result["trackingEvaluationResults"][trackingMethod][
                        "geometricErrors"
                    ]["accumulated"]
                ),
                "reprojectionError": np.mean(
                    result["trackingEvaluationResults"][trackingMethod][
                        "reprojectionErrors"
                    ]["mean"]
                ),
                "runtime": runtimeScaleFactor
                * np.mean(
                    result["trackingEvaluationResults"][trackingMethod][
                        "runtimeResults"
                    ]["runtimesPerIteration"]
                ),
            }

        dataSetTableRow = f"""----------Cut-------------
            \multirow{{{dataSetTableRowEntries["n_methods"]}}}{{*}}{{${dataSetTableRowEntries["modelID"]}$}} & \multirow{{{dataSetTableRowEntries["n_methods"]}}}{{*}}{{${dataSetTableRowEntries["n_tracked_frames"]}$}} & \multirow{{{dataSetTableRowEntries["n_methods"]}}}{{*}}{{${dataSetTableRowEntries["n_labeld_frames"]}$}} & \\acs{{CPD}} & ${dataSetTableRowEntries["cpd"]["trackingError"]:.2f}$ & ${dataSetTableRowEntries["cpd"]["geometricError"]:.2f}$ & ${int(dataSetTableRowEntries["cpd"]["reprojectionError"])}$ & ${dataSetTableRowEntries["cpd"]["runtime"]:.2f}$ \\\\
            &  &  & \\acs{{SPR}} & ${dataSetTableRowEntries["spr"]["trackingError"]:.2f}$ & ${dataSetTableRowEntries["spr"]["geometricError"]:.2f}$ & ${int(dataSetTableRowEntries["spr"]["reprojectionError"])}$ & ${dataSetTableRowEntries["spr"]["runtime"]:.2f}$ \\\\
            &  &  & \\acs{{KPR}} & ${dataSetTableRowEntries["krcpd"]["trackingError"]:.2f}$ & ${dataSetTableRowEntries["krcpd"]["geometricError"]:.2f}$ & ${int(dataSetTableRowEntries["krcpd"]["reprojectionError"])}$ & ${dataSetTableRowEntries["krcpd"]["runtime"]:.2f}$ \\\\
            ----------Cut-------------"""
        print(dataSetTableRow)
    return


def createTrackingErrorTimeSeriesPlots(results):
    for dataSetResult in results:
        createTrackingErrorTimeSeriesPlot(dataSetResult, highlightFrames=[1, 10, 22])


def createTrackingErrorTimeSeriesPlot(
    dataSetResult, lineColors=None, highlightFrames=None, highlightColor=[1, 0, 0]
):
    """creates for a data set a folder with a tracking error plot over all methods, and images for each method for certain frames which can be custumized and are indicated as vertial lines in the plot"""
    highlightFrames = [] if highlightFrames is None else highlightFrames
    fig = plt.figure()
    ax = plt.axes()
    trackingErrorLines = []
    for key in dataSetResult["trackingEvaluationResults"]:
        trackingErrors = dataSetResult["trackingEvaluationResults"][key][
            "trackingErrors"
        ]
        (trackingErrorLine,) = ax.plot(list(range(len(trackingErrors))), trackingErrors)
        trackingErrorLine.set_label(key)
        trackingErrorLines.append(trackingErrorLine)
    for highlightFrame in highlightFrames:
        ax.axvline(x=highlightFrame, color=highlightColor)
    ax.legend()

    # save time series plot under data/eval/tracking/imgs/trackingErrorTimeSeries/<dataSetName>
    dataSetPath = dataSetResult["dataSetPath"]
    # make folder if it does not exist
    saveRootFolderPath = savePathsForRepresentations["trackingErrorTimeSeries"]
    if not os.path.exists(saveRootFolderPath):
        os.makedirs(saveRootFolderPath)
    # make folder for dataSet
    dataSetName = dataSetPath.split("/")[-2]
    saveFolderPath = os.path.join(saveRootFolderPath, dataSetName)
    if not os.path.exists(saveFolderPath):
        os.makedirs(saveFolderPath)
    fileName = "trackingErrorTimeSeries"
    savePath = os.path.join(saveFolderPath, fileName)
    # save as png
    plt.savefig(savePath)
    tikzplotlib.save(savePath + ".tex")
    # save as tixfigure

    # save highlight frames under data/eval/tracking/imgs/trackingErrorTimeSeries/<dataSetName>/<method> # raw, cpd, spr, krp ,...
    folders = ["raw"] + list(dataSetResult["trackingEvaluationResults"].keys())
    for subfolder in folders:
        subfolderPath = os.path.join(saveFolderPath, subfolder)
        if not os.path.exists(subfolderPath):
            os.makedirs(subfolderPath)
        for i, highlightFrame in enumerate(highlightFrames):
            rgbImg = eval.getImage(highlightFrame, dataSetPath)
            fileName2DPlot = "hightligth_img_" + str(i + 1)
            fileName3DPlot = "hightligth_3Dplot_" + str(i + 1)
            savePath2DPlot = os.path.join(subfolderPath, fileName2DPlot)
            savePath3DPlot = os.path.join(subfolderPath, fileName3DPlot)
            if subfolder == "raw":
                eval.saveImage(rgbImg, savePath2DPlot)
            else:
                # plot result in 2D
                positions3D = dataSetResult["trackingEvaluationResults"][subfolder][
                    "trackingResult"
                ]["registrations"][highlightFrame]["T"]
                adjacencyMatrix = dataSetResult["trackingEvaluationResults"][subfolder][
                    "trackingResult"
                ]["adjacencyMatrix"]
                rgbImg = eval.plotTrackingResult2D(
                    frame=highlightFrame,
                    dataSetPath=dataSetPath,
                    positions3D=positions3D,
                    adjacencyMatrix=adjacencyMatrix,
                )
                eval.saveImage(rgbImg, savePath2DPlot)
                # plot result in 3D
                pointCloud = dataSetResult["trackingEvaluationResults"][subfolder][
                    "trackingResult"
                ]["registrations"][highlightFrame]["Y"]
                fig, ax = eval.plotTrackingResult3D(
                    pointCloud=pointCloud,
                    targets=positions3D,
                    adjacencyMatrix=adjacencyMatrix,
                    pointCloudColor=[1, 0, 0],
                    targetColor=[0, 0, 1],
                    lineColor=[0, 0, 1],
                    pointCloudPointSize=1,
                    targetPointSize=10,
                    pointCloudAlpha=0.1,
                    targetAlpha=1,
                    axisLimX=[0.15, 0.65],
                    axisLimY=[0.0, 0.5],
                    axisLimZ=[0.25, 0.75],
                    elevation=25,
                    azimuth=70,
                )
                plt.savefig(
                    savePath3DPlot + ".png",
                )
                # plot result in DART
                # eval.visualizeConfigurationInDart()
    return


if __name__ == "__main__":
    # setup eval class
    if resultsToLoad[0] == -1:
        resultsToEvaluate = resultFolderPaths
    else:
        resultsToEvaluate = [
            resultFolderPath
            for i, resultFolderPath in enumerate(resultFolderPaths)
            if i in resultsToLoad
        ]

    accumulatedResults = []
    for resultFolderPath in resultsToEvaluate:
        # set file paths
        dataSetName = resultFolderPath.split("/")[-1]
        resultFilePath = resultFolderPath + "/" + resultFileName + ".pkl"

        # select config
        configFilePath = "/evalConfigs/evalConfig" + "_" + dataSetName + ".json"

        # setup evalulation
        global eval
        pathToConfigFile = os.path.dirname(os.path.abspath(__file__)) + configFilePath
        eval = TrackingEvaluation(configFilePath=pathToConfigFile)

        # load result for the data set
        dataSetResult = eval.loadResults(resultFilePath)
        accumulatedResults.append(dataSetResult)

    # create latex table with results
    if controlOptions["tabularizeResults"]:
        tabularizeResults(accumulatedResults)

    # create tracking error time series
    if controlOptions["createTrackingTimeSeriesPlots"]:
        createTrackingErrorTimeSeriesPlots(accumulatedResults)

    # create reprojection error time series

    # create image sequence for mantipulation sceanrios
