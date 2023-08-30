import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy.spatial import distance_matrix

try:
    sys.path.append(os.getcwd().replace("/eval", ""))
    from src.evaluation.tracking.trackingEvaluation import TrackingEvaluation
    from src.visualization.plot2D import *
except:
    print("Imports for plotting script tracking error time series failed.")
    raise

global eval
eval = TrackingEvaluation()

controlOpt = {
    "resultsToLoad": [0, 1, 2],
    "methods": ["cpd", "spr", "kpr", "krcpd"],  # "cpd", "spr", "kpr", "krcpd"
    "frames": [5, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650],
    "save": True,
    "showPlot": False,
    "saveFolder": "data/eval/tracking/plots/trackingResults2D",
    "saveName": "trackingResult2D",
    "´postProcessingToAlginPointCloud": False,
}
styleOpt = {
    "lineColor": [0, 81 / 255, 158 / 255],
    "lineThickness": 5,
    "circleColor": [0, 81 / 255, 158 / 255],
    "circleRadius": 10,
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
    dataSetPath = dataSetResult["dataSetPath"]
    adjacencyMatrix = dataSetResult["trackingResults"][method]["adjacencyMatrix"]
    T = eval.findCorrespondingEntryFromKeyValuePair(
        dataSetResult["trackingResults"][method]["registrations"], "frame", frame
    )["T"]
    Y = eval.findCorrespondingEntryFromKeyValuePair(
        dataSetResult["trackingResults"][method]["registrations"], "frame", frame
    )["Y"]

    if controlOpt["´postProcessingToAlginPointCloud"]:
        D = distance_matrix(T, Y)
        C = np.argmin(D, axis=0)
        N = len(T)
        predictedPositions = T.copy()
        for n in range(0, N):
            incides = np.where(C == n)[0]
            if len(incides) > 3:
                Yc = Y[incides, :]
                predictedPositions[n, :] = np.mean(Yc, axis=0)
        positions3D = predictedPositions
    else:
        positions3D = T

    resultImg = eval.plotTrackingResult2D(
        frame,
        dataSetPath,
        positions3D,
        adjacencyMatrix,
        lineColor=styleOpt["lineColor"],
        circleColor=styleOpt["circleColor"],
        lineThickness=styleOpt["lineThickness"],
        circleRadius=styleOpt["circleRadius"],
    )
    # display Visualization
    if controlOpt["showPlot"]:
        eval.plotImageWithMatplotlib(resultImg, block=True)
    # save Visualization
    if controlOpt["save"]:
        filename = controlOpt["saveName"] + "_frame_" + str(frame)
        dataSetName = dataSetPath.split("/")[-2]
        folderPath = os.path.join(controlOpt["saveFolder"], dataSetName, method)
        if not os.path.exists(folderPath):
            os.makedirs(folderPath, exist_ok=True)
        savePath = os.path.join(folderPath, filename)
        eval.saveImage(resultImg, savePath)
        if verbose:
            print("Saved frame {} for method {}.".format(frame, method))
    return None


if __name__ == "__main__":
    # load all results
    results = []
    for resultFilePath in [resultFolderPaths[x] for x in controlOpt["resultsToLoad"]]:
        resultFilePath = os.path.join(resultFilePath, resultFileName)
        result = loadResult(resultFilePath)
        results.append(result)
    # create plot
    for result in results:
        for method in controlOpt["methods"]:
            for frame in controlOpt["frames"]:
                createPlots(result, frame, method)
