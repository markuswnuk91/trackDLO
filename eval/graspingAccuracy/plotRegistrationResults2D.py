import sys
import os
import matplotlib.pyplot as plt
import numpy as np

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
    "showPlot": False,
    "save": True,
    "verbose": True,
}

saveOpt = {
    "saveFileName": "registrationResult2D",
    "saveFolder": "data/eval/graspingAccuracy/plots/registrationResults2D",
    "saveName": "result",
}

styleOpt = {
    "colorPalette": thesisColorPalettes["viridis"],
    "lineThickness": 5,
    "circleRadius": 10,
}

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


def plotBranchWiseColoredRegistrationResult2D(
    result,
    method,
    num,
    colorPalette=None,
    lineThickness=None,
    circleRadius=None,
):
    colorPalette = (
        thesisColorPalettes["viridis"] if colorPalette is None else colorPalette
    )
    lineThickness = 5 if lineThickness is None else lineThickness
    circleRadius = 10 if circleRadius is None else circleRadius

    trackingResult = result["trackingResults"][method]
    registrationResult = result["trackingResults"][method]["registrationResults"][num]
    frame = registrationResult["frame"]
    dataSetPath = result["dataSetPath"]
    rgbImg = eval.getDataSet(frame, dataSetPath)[0]  # load image

    adjacencyMatrix = trackingResult["adjacencyMatrix"]
    positions3D = registrationResult["result"]["T"]
    positions2D = eval.reprojectFrom3DRobotBase(positions3D, dataSetPath)
    B = trackingResult["B"]
    numBranches = len(set(B))

    colorScaleCoordinates = np.linspace(0, 1, numBranches)
    branchColors = []
    for s in colorScaleCoordinates:
        branchColors.append(colorPalette.to_rgba(s)[:3])

    branchNodeIndices = np.where(np.sum(adjacencyMatrix, axis=1) >= 3)[0]
    for branchIndex in range(0, numBranches):
        indices = np.where(np.array(B) == branchIndex)[0]
        # add indices of adjacent branches
        for branchNodeIndex in branchNodeIndices:
            adjacentNodeIndices = np.where(adjacencyMatrix[branchNodeIndex, :] != 0)[0]
            for adjacentNodeIndex in adjacentNodeIndices:
                if (B[adjacentNodeIndex] == branchIndex) and (
                    not (branchNodeIndex in indices)
                ):
                    indices = np.append(indices, branchNodeIndex)
        branchPositions = positions2D[indices, :]
        branchAdjacencyMatrix = np.array(
            [[adjacencyMatrix[row][col] for col in indices] for row in indices]
        )
        rgbImg = plotGraph2_CV(
            rgbImg=rgbImg,
            positions2D=branchPositions,
            adjacencyMatrix=branchAdjacencyMatrix,
            lineColor=branchColors[branchIndex],
            circleColor=branchColors[branchIndex],
            lineThickness=lineThickness,
            circleRadius=circleRadius,
        )
        for branchNodeIndex in branchNodeIndices:
            circleColor = tuple([x * 255 for x in branchColors[B[branchNodeIndex]]])
            cv2.circle(
                rgbImg,
                positions2D[branchNodeIndex, :],
                circleRadius,
                circleColor,
                thickness=-1,
            )
    return rgbImg


if __name__ == "__main__":
    if controlOpt["resultsToLoad"][0] == -1:
        resultsToEvaluate = resultFolderPaths
    else:
        resultsToEvaluate = [
            resultFolderPath
            for i, resultFolderPath in enumerate(resultFolderPaths)
            if i in controlOpt["resultsToLoad"]
        ]

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
                registrationResultsToEvaluate = list(range(0, numRegistrationResults))
            else:
                registrationResultsToEvaluate = controlOpt[
                    "registrationResultsToEvaluate"
                ]
            for nRegistrationResult in registrationResultsToEvaluate:
                colorPalette = styleOpt["colorPalette"]
                lineThickness = styleOpt["lineThickness"]
                circleRadius = styleOpt["circleRadius"]
                rgbImg = plotBranchWiseColoredRegistrationResult2D(
                    result,
                    method,
                    nRegistrationResult,
                    colorPalette=colorPalette,
                    lineThickness=lineThickness,
                    circleRadius=circleRadius,
                )
                if controlOpt["showPlot"]:
                    eval.plotImageWithMatplotlib(
                        rgbImg, title="registration result " + method, block=True
                    )

                if controlOpt["save"]:
                    dataSetName = result["dataSetName"]
                    fileID = "_".join(
                        result["trackingResults"][method]["registrationResults"][
                            nRegistrationResult
                        ]["fileName"].split("_")[0:3]
                    )
                    fileName = fileID + "_" + saveOpt["saveName"]
                    saveFolderPath = saveOpt["saveFolder"]
                    saveFolderPath = os.path.join(saveFolderPath, dataSetName, method)
                    saveFilePath = os.path.join(saveFolderPath, fileName)
                    if not os.path.exists(saveFolderPath):
                        os.makedirs(saveFolderPath, exist_ok=True)
                    eval.saveImage(rgbImg, saveFilePath)
                    if controlOpt["verbose"]:
                        print(
                            "Saved registration {}/{} from method {}/{} of result {}/{} at {}".format(
                                nRegistrationResult + 1,
                                len(registrationResultsToEvaluate),
                                nMethod + 1,
                                len(methodsToEvaluate),
                                nResult + 1,
                                len(resultsToEvaluate),
                                saveFilePath,
                            )
                        )
    if controlOpt["verbose"]:
        print("Finished result generation.")
