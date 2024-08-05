import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy.spatial import distance_matrix

try:
    sys.path.append(os.getcwd().replace("/eval", ""))
    from src.evaluation.tracking.trackingEvaluation import TrackingEvaluation
    from src.visualization.plot3D import *
except:
    print("Imports for plotting script tracking error time series failed.")
    raise

global eval
eval = TrackingEvaluation()

controlOpt = {
    "resultsToLoad": [0],
    "methods": ["cpd", "spr", "krcpd"],  # "cpd", "spr", "kpr", "krcpd"
    "frames": [[0, 5, 100, 250, 400, 500, 650]],
    "save": False,
    "saveAsPGF": False,
    "showPlot": True,
    "saveFolder": "data/eval/tracking/plots/trackingResults3D",
    "saveName": "trackingResult3D",
    "dpi": 100,
}
layoutOpt = {
    "lineColor": [0, 81 / 255, 158 / 255],
    "lineWidth": 1.5,
    "circleColor": [0, 81 / 255, 158 / 255],
    "circleRadius": 10,
    "pointCloudColor": [1, 0, 0],
    "targetColor": [0, 0, 1],
    "pointCloudPointSize": 1,
    "targetPointSize": 10,
    "pointCloudAlpha": 0.1,
    "targetAlpha": 1,
    "elevation": 30,  # 30 for isometric view
    "azimuth": 45,  # 45 for isometric view
    "axLimX": [0.3, 0.7],
    "axLimY": [0.1, 0.6],
    "axLimZ": [0.35, 0.75],
}
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


def createPlots(dataSetResult, frame, method, verbose=True):
    fig, ax = setupLatexPlot3D()

    registrationResult = eval.findCorrespondingEntryFromKeyValuePair(
        dataSetResult["trackingResults"][method]["registrations"], "frame", frame
    )
    pointCloud = registrationResult["Y"]
    targets = registrationResult["T"]
    adjacencyMatrix = dataSetResult["trackingResults"][method]["adjacencyMatrix"]
    dataSetPath = dataSetResult["dataSetPath"]
    # TODO: use plotBranchWiseColoredTopology3D function for plotting
    # ax = eval.plotTrackingResult3D(
    #     ax=ax,
    #     pointCloud=pointCloud,
    #     targets=targets,
    #     adjacencyMatrix=adjacencyMatrix,
    #     pointCloudColor=[1, 0, 0],
    #     targetColor=[0, 0, 1],
    #     lineColor=layoutOpt["lineColor"],
    #     pointCloudPointSize=layoutOpt["pointCloudPointSize"],
    #     targetPointSize=layoutOpt["targetPointSize"],
    #     pointCloudAlpha=layoutOpt["pointCloudAlpha"],
    #     targetAlpha=layoutOpt["targetAlpha"],
    #     elevation=layoutOpt["elevation"],
    #     azimuth=layoutOpt["azimuth"],
    #     lineWidth=layoutOpt["lineWidth"],
    # )
    modelParameters = dataSetResult["trackingResults"][method]["modelParameters"]
    topologyModel = eval.generateModel(modelParameters)
    ax = eval.plotBranchWiseColoredTrackingResult3D(
        ax=ax, X=targets, topology=topologyModel
    )
    # customize figure
    ax.axes.set_xlim3d(left=layoutOpt["axLimX"][0], right=layoutOpt["axLimX"][1])
    ax.axes.set_ylim3d(bottom=layoutOpt["axLimY"][0], top=layoutOpt["axLimY"][1])
    ax.axes.set_zlim3d(bottom=layoutOpt["axLimZ"][0], top=layoutOpt["axLimZ"][1])

    # save figure
    if controlOpt["save"]:
        fileName = controlOpt["saveName"] + "_frame_" + str(frame)
        dataSetName = dataSetPath.split("/")[-2]
        folderPath = os.path.join(controlOpt["saveFolder"], dataSetName, method)
        if not os.path.exists(folderPath):
            os.makedirs(folderPath, exist_ok=True)
        filePath = os.path.join(folderPath, fileName)
        if controlOpt["saveAsPGF"]:
            raise NotImplementedError
            # plt.savefig(filePath, format="pgf", bbox_inches="tight", pad_inches=0)
        else:
            plt.savefig(
                filePath, dpi=controlOpt["dpi"], bbox_inches="tight", pad_inches=0
            )
        if verbose:
            print("Saved frame {} for method {}.".format(frame, method))
    # display figure
    if controlOpt["showPlot"]:
        plt.show(block=True)
    else:
        plt.close(fig)
    return None


if __name__ == "__main__":
    # load all results
    results = []
    for resultFilePath in [resultFolderPaths[x] for x in controlOpt["resultsToLoad"]]:
        resultFilePath = os.path.join(resultFilePath, resultFileName)
        result = loadResult(resultFilePath)
        results.append(result)
    # create plot
    for i, result in enumerate(results):
        for method in controlOpt["methods"]:
            for frame in controlOpt["frames"][i]:
                createPlots(result, frame, method)
