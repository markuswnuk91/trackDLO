import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib
import cv2

try:
    sys.path.append(os.getcwd().replace("/eval", ""))
    from src.evaluation.tracking.trackingEvaluation import (
        TrackingEvaluation,
    )
    from src.visualization.dartVisualizer import DartVisualizer, DartScene
    from src.visualization.plot3D import *
    from src.visualization.plot2D import *
except:
    print("Imports for Tracking Result Evaluation failed.")
    raise


# script control parameters
controlOptions = {
    "tabularizeResults": True,
    "createTrackingTimeSeriesPlots": True,
    "createStackedImagesPlot": True,
    "createReprojectionErrorBoxPlot": True,
}
resultFileName = "result"
resultFolderPaths = [
    "data/eval/tracking/results/20230524_171237_ManipulationSequences_mountedWireHarness_modelY",
    "data/eval/tracking/results/20230807_162939_ManipulationSequences_mountedWireHarness_partial",
    "data/eval/tracking/results/20230524_161235_ManipulationSequences_mountedWireHarness_arena",
]
resultsToLoad = [1]
savePathsForRepresentations = {
    "trackingErrorTimeSeries": "data/eval/tracking/trackingTimeSeriesPlots"
}
highlightFrames = [1, 10, 20, 30, 50, 70, 100, 150, 200]


def tabularizeResults(accumulatedResults, methodsToPrint=["cpd", "spr", "krcpd"]):
    trackingErrorScaleFactor = 100  # cm
    geometricErrorScaleFactor = 100  # cm
    runtimeScaleFactor = 1000  # mm
    for result in accumulatedResults:
        modelID = eval.getModelID(
            result["initializationResult"]["modelParameters"]["modelInfo"]["name"]
        )
        trackingMethods = list(result["trackingEvaluationResults"].keys())
        dataSetTableRowEntries = {
            "modelID": modelID,
            "n_tracked_frames": len(
                result["trackingResults"][trackingMethods[0]]["frames"]
            ),
            "n_methods": len(methodsToPrint),
        }
        if (
            "reprojectionErrors"
            in result["trackingEvaluationResults"][trackingMethods[0]]
        ):
            dataSetTableRowEntries["n_labeled_frames"] = (
                len(
                    result["trackingEvaluationResults"][trackingMethods[0]][
                        "reprojectionErrors"
                    ]["frames"]
                ),
            )
        else:
            dataSetTableRowEntries["n_labeled_frames"] = 0
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
                "runtime": runtimeScaleFactor
                * np.mean(
                    result["trackingEvaluationResults"][trackingMethod][
                        "runtimeResults"
                    ]["runtimesPerIteration"]
                ),
            }

            if (
                "reprojectionErrors"
                in result["trackingEvaluationResults"][trackingMethods[0]]
            ):
                dataSetTableRowEntries[trackingMethod]["reprojectionError"] = np.mean(
                    result["trackingEvaluationResults"][trackingMethod][
                        "reprojectionErrors"
                    ]["mean"]
                )
            else:
                dataSetTableRowEntries[trackingMethod]["reprojectionError"] = 0

        dataSetTableRow = f"""----------Cut-------------
            \multirow{{{dataSetTableRowEntries["n_methods"]}}}{{*}}{{${dataSetTableRowEntries["modelID"]}$}} & \multirow{{{dataSetTableRowEntries["n_methods"]}}}{{*}}{{${dataSetTableRowEntries["n_tracked_frames"]}$}} & \multirow{{{dataSetTableRowEntries["n_methods"]}}}{{*}}{{${dataSetTableRowEntries["n_labeled_frames"]}$}} & \\acs{{CPD}} & ${dataSetTableRowEntries["cpd"]["trackingError"]:.2f}$ & ${dataSetTableRowEntries["cpd"]["geometricError"]:.2f}$ & ${int(dataSetTableRowEntries["cpd"]["reprojectionError"])}$ & ${dataSetTableRowEntries["cpd"]["runtime"]:.2f}$ \\\\
            &  &  & \\acs{{SPR}} & ${dataSetTableRowEntries["spr"]["trackingError"]:.2f}$ & ${dataSetTableRowEntries["spr"]["geometricError"]:.2f}$ & ${int(dataSetTableRowEntries["spr"]["reprojectionError"])}$ & ${dataSetTableRowEntries["spr"]["runtime"]:.2f}$ \\\\
            &  &  & \\acs{{KPR}} & ${dataSetTableRowEntries["krcpd"]["trackingError"]:.2f}$ & ${dataSetTableRowEntries["krcpd"]["geometricError"]:.2f}$ & ${int(dataSetTableRowEntries["krcpd"]["reprojectionError"])}$ & ${dataSetTableRowEntries["krcpd"]["runtime"]:.2f}$ \\\\
            ----------Cut-------------"""
        print(dataSetTableRow)
    return


def createTrackingErrorTimeSeriesPlots(results):
    for dataSetResult in results:
        createTrackingErrorTimeSeriesPlot(
            dataSetResult, highlightFrames=highlightFrames
        )


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
            fileNameDartPlot = "hightligth_dart_" + str(i + 1)
            fileNameDartFrontViewPlot = "hightligth_dart_front_" + str(i + 1)
            savePath2DPlot = os.path.join(subfolderPath, fileName2DPlot)
            savePath3DPlot = os.path.join(subfolderPath, fileName3DPlot)
            savePathDartPlot = os.path.join(subfolderPath, fileNameDartPlot)
            savePathDartFrontViewPlot = os.path.join(
                subfolderPath, fileNameDartFrontViewPlot
            )
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
                modelParameters = dataSetResult["trackingResults"][
                    list(dataSetResult["trackingResults"].keys())[0]
                ]["modelParameters"]
                model = eval.generateModel(modelParameters)

                q = model.computeInverseKinematics(
                    targets=positions3D, damping=0.5, numIterations=100, verbose=True
                )
                dartScene = DartScene(model.skel, q)
                dartScene.saveFrame(
                    savePathDartPlot,
                    grid=True,
                    x=0,
                    y=0,
                    width=1000,
                    height=1000,
                    eye=[0.5, 1.3, 1.0],
                    center=[0.5, 0, 0.3],
                    up=[0, -1, 0],
                    format=".png",
                )
                dartScene.saveFrame(
                    savePathDartFrontViewPlot,
                    grid=True,
                    x=0,
                    y=0,
                    width=1000,
                    height=1000,
                    eye=[4, 0, 1.5],
                    center=[0.5, 0, 0.3],
                    up=[0, 0, 1],
                    format=".png",
                )
                # qDart = dartScene.computeCartesianForceInverseKinematics(
                #     model.skel, targetPositions=positions3D, vis=True, verbose=True
                # )
                # print(qDart)
                # eval.visualizeConfigurationInDart()
    return


def createStackedImagesPlot(results):
    # load images

    dataSetPath = results[0]["dataSetPath"]
    numFrames = eval.getNumImageSetsInDataSet(dataSetPath)
    frames = list(range(0, numFrames))
    nthFrame = 100
    frames = frames[1::nthFrame]
    images = []
    for frame in frames:
        image = eval.getImage(frame, dataSetPath)
        images.append(image)

    # Define the alpha value
    alpha = 0.5
    accumulated_blend = images[0]

    for img in images[1:]:
        accumulated_blend = cv2.addWeighted(accumulated_blend, 0.5, img, 0.5, 0)

    cv2.imshow("Blended Image", accumulated_blend)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def createReprojectionErrorBoxPlots(accumulatedResults):
    for result in accumulatedResults:
        createReprojectionErrorBoxPlot(result)
    return


def createReprojectionErrorBoxPlot(result, methods=["cpd", "spr", "krcpd"]):
    # Define the width of the bars
    width = 0.2

    means = []
    stds = []
    positions = []
    keys = [key for key in result["trackingEvaluationResults"].keys()]
    labeledFrames = result["trackingEvaluationResults"][keys[0]]["reprojectionErrors"][
        "frames"
    ]

    X = labeledFrames
    for i, method in enumerate(methods):
        # Sample mean and std results
        reprojectionErrorsPerFrame = result["trackingEvaluationResults"]["cpd"][
            "reprojectionErrors"
        ]["reprojectionErrors"]
        mean = []
        std = []
        for reprojectionErrors in reprojectionErrorsPerFrame:
            meanPerFrame = np.mean(reprojectionErrors)
            stdPerFrame = np.std(reprojectionErrors)
            mean.append(meanPerFrame)
            std.append(stdPerFrame)
        means.append(mean)
        stds.append(std)

        # Define the positions for each method's bars
        sign = np.sign((((i + 1) - (len(methods) + 1) / 2)))
        shif = sign * np.abs(((i + 1) - (len(methods) + 1) / 2)) * width
        position = np.array(X) + shif
        positions.append(position)

    # plot error bars
    for i, method in enumerate(methods):
        plt.errorbar(
            positions[i],
            means[i],
            yerr=stds[i],
            fmt="o",
            label=method,
            capsize=5,
        )

    # Set the title, labels, and a legend
    plt.xlabel("frames")
    plt.ylabel("Results")
    plt.xticks(X)
    plt.legend()

    # Display the plot
    plt.tight_layout()
    plt.show()
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

    # create stacked image plot
    if controlOptions["createStackedImagesPlot"]:
        createStackedImagesPlot(accumulatedResults)

    # create reprojection error time series / bar plot ?
    if controlOptions["createReprojectionErrorBoxPlot"]:
        createReprojectionErrorBoxPlots(accumulatedResults)
    # create image sequence for mantipulation sceanrios
