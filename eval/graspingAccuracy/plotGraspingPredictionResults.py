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
    "saveFolder": "data/eval/graspingAccuracy/plots/graspingPredictionResult",
    "saveFileName": "graspingPrediction",
}

styleOpt = {
    "plotRegistrationResult": True,
    "plotGroundTruth": True,
    "groundTruthColor": [0, 1, 0],
    "predictionColor": [1, 0, 0],
    "gipperWidth3D": 0.15,
    "fingerWidth2D": 0.5,
    "centerThickness": 12,
    "lineThickness": 8,
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


def plotPredictedGraspingPose(
    result,
    method,
    num,
    frame_offset=1,  # frame offset between registration frame and robot state frame
):
    registrationResult = result["trackingResults"][method]["registrationResults"][num]
    frame = registrationResult["frame"]
    frame_grasp = frame + frame_offset
    dataSetPath = result["dataSetPath"]

    # ground truth
    (
        groundTruthGraspingPose,
        groundTruthGraspingPosition,
        groundTruthGraspingRotationMatrix,
    ) = eval.loadGroundTruthGraspingPose(
        dataSetPath, frame_grasp
    )  # ground truth grasping position is given by the frame after the prediction frame
    groundTruthGraspingAxis = groundTruthGraspingRotationMatrix[:3, 0]
    # prediction
    graspingLocalCoordinates = eval.loadGraspingLocalCoordinates(dataSetPath)
    graspingLocalCoordinate = graspingLocalCoordinates[num]
    T = registrationResult["result"]["T"]
    B = result["trackingResults"][method]["B"]
    S = result["initializationResult"]["localizationResult"]["SInit"]
    (
        predictedGraspingPosition,
        predictedGraspingAxis,
    ) = eval.predictGraspingPositionAndAxisFromRegistrationTargets(
        T, B, S, graspingLocalCoordinate
    )

    if styleOpt["plotGroundTruth"]:
        graspingPositions3D = np.vstack(
            (groundTruthGraspingPosition, predictedGraspingPosition)
        )
        graspingAxes3D = np.vstack((groundTruthGraspingAxis, predictedGraspingAxis))
        colors = [styleOpt["groundTruthColor"], styleOpt["predictionColor"]]
    else:
        graspingPositions3D = predictedGraspingPosition
        graspingAxes3D = predictedGraspingAxis
        colors = [styleOpt["predictionColor"]]
    gipperWidth3D = styleOpt["gipperWidth3D"]
    fingerWidth2D = styleOpt["fingerWidth2D"]
    centerThickness = styleOpt["centerThickness"]
    lineThickness = styleOpt["lineThickness"]

    rgbImg = eval.getDataSet(frame_grasp, dataSetPath)[0]

    positions2D = eval.reprojectFrom3DRobotBase(T, dataSetFolderPath=dataSetPath)
    adjacencyMatrix = result["trackingResults"][method]["adjacencyMatrix"]
    if styleOpt["plotRegistrationResult"]:
        rgbImg = eval.plotBranchWiseColoredRegistrationResult(
            rgbImg, positions2D, adjacencyMatrix, B
        )

    # reproject grasping positions in image
    graspingPositions2D = eval.reprojectFrom3DRobotBase(
        graspingPositions3D, dataSetPath
    )
    # reproject grasping axes in image
    graspingAxesStartPoints3D = graspingPositions3D - gipperWidth3D / 2 * graspingAxes3D
    graspingAxesEndPoints3D = graspingPositions3D + gipperWidth3D / 2 * graspingAxes3D
    graspingAxesStartPoints2D = eval.reprojectFrom3DRobotBase(
        graspingAxesStartPoints3D, dataSetPath
    )
    graspingAxesEndPoints2D = eval.reprojectFrom3DRobotBase(
        graspingAxesEndPoints3D, dataSetPath
    )
    # 2D grasping axes
    graspingAxes2D = graspingAxesEndPoints2D - graspingAxesStartPoints2D
    rgbImg = eval.drawGraspingPoses2D(
        rgbImage=rgbImg,
        graspingPositions2D=graspingPositions2D,
        graspingAxes2D=graspingAxes2D,
        colors=colors,
        fingerWidth2D=fingerWidth2D,
        centerThickness=centerThickness,
        lineThickness=lineThickness,
        markerFill=-1,
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
                registrationResultsToEvaluate = list(
                    range(
                        0, numRegistrationResults - 1
                    )  # do not evaluate last registration result since this is only the final frame
                )
            else:
                registrationResultsToEvaluate = controlOpt[
                    "registrationResultsToEvaluate"
                ]
            for nRegistrationResult in registrationResultsToEvaluate:
                rgbImg = plotPredictedGraspingPose(
                    result,
                    method,
                    nRegistrationResult,
                )
                if controlOpt["showPlot"]:
                    eval.plotImageWithMatplotlib(
                        rgbImg, title="grasping prediction " + method
                    )
                    plt.show(block=True)

                if controlOpt["save"]:
                    dataSetName = result["dataSetName"]
                    fileID = "_".join(
                        result["trackingResults"][method]["registrationResults"][
                            nRegistrationResult
                        ]["fileName"].split("_")[0:3]
                    )
                    # fileName = fileID + "_" + saveOpt["saveFileName"]
                    fileName = (
                        saveOpt["saveFileName"] + "_grasp_" + str(nRegistrationResult)
                    )
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
