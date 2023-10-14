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
    "saveManipulationImg": True,
    "showPlot": False,
    "save": True,
    "verbose": True,
}

saveOpt = {
    "saveFolder": "data/eval/graspingAccuracy/plots/combinedGraspingPredictionResult",
    "saveFileNames": {"prediction": "predictionResult", "manipulation": "grasp"},
}

styleOpt = {
    "groundTruthColor": thesisColors["uniSLightBlue"],
    "predictionColors": {
        "cpd": thesisColors["susieluMagenta"],
        "spr": thesisColors["susieluGold"],
        "kpr": thesisColors["susieluBlue"],
    },
    "gipperWidth3D": 0.1,
    "fingerWidth2D": 0.5,
    "centerThickness": 10,
    "lineThickness": 5,
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
    rgbImg,
    graspingPositions2D,
    graspingAxes2D,
    predictionColor=None,
    groundTruthColor=None,
):
    predictionColor = (
        styleOpt["predictionColor"] if predictionColor is None else predictionColor
    )
    groundTruthColor = (
        styleOpt["groundTruthColor"] if groundTruthColor is None else groundTruthColor
    )
    colors = [groundTruthColor, predictionColor]

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
        numRegistrationResults = eval.getNumRegistrationResults(result)
        if controlOpt["registrationResultsToEvaluate"][0] == -1:
            registrationResultsToEvaluate = list(
                range(
                    0, numRegistrationResults - 1
                )  # do not evaluate last registration result since this is only the final frame
            )
        else:
            registrationResultsToEvaluate = controlOpt["registrationResultsToEvaluate"]
        for nRegistrationResult in registrationResultsToEvaluate:
            for nMethod, method in enumerate(methodsToEvaluate):
                registrationResult = result["trackingResults"][method][
                    "registrationResults"
                ][nRegistrationResult]
                frame = registrationResult["frame"]
                dataSetPath = result["dataSetPath"]
                if nMethod == 0:
                    rgbImg = eval.getDataSet(frame, dataSetPath)[0]

                gipperWidth3D = styleOpt["gipperWidth3D"]
                fingerWidth2D = styleOpt["fingerWidth2D"]
                centerThickness = styleOpt["centerThickness"]
                lineThickness = styleOpt["lineThickness"]
                # ground truth
                (
                    groundTruthGraspingPose,
                    groundTruthGraspingPosition,
                    groundTruthGraspingRotationMatrix,
                ) = eval.loadGroundTruthGraspingPose(
                    dataSetPath, frame + 1
                )  # ground truth grasping position is given by the frame after the prediction frame
                groundTruthGraspingAxis = groundTruthGraspingRotationMatrix[:3, 0]
                # prediction
                graspingLocalCoordinates = eval.loadGraspingLocalCoordinates(
                    dataSetPath
                )
                graspingLocalCoordinate = graspingLocalCoordinates[nRegistrationResult]
                T = registrationResult["result"]["T"]
                B = result["trackingResults"][method]["B"]
                S = result["initializationResult"]["localizationResult"]["SInit"]
                (
                    predictedGraspingPosition,
                    predictedGraspingAxis,
                ) = eval.predictGraspingPositionAndAxisFromRegistrationTargets(
                    T, B, S, graspingLocalCoordinate
                )
                graspingPositions3D = np.vstack(
                    (groundTruthGraspingPosition, predictedGraspingPosition)
                )
                graspingAxes3D = np.vstack(
                    (groundTruthGraspingAxis, predictedGraspingAxis)
                )
                positions2D = eval.reprojectFrom3DRobotBase(
                    T, dataSetFolderPath=dataSetPath
                )
                # reproject grasping positions in image
                graspingPositions2D = eval.reprojectFrom3DRobotBase(
                    graspingPositions3D, dataSetPath
                )
                # reproject grasping axes in image
                graspingAxesStartPoints3D = (
                    graspingPositions3D - gipperWidth3D / 2 * graspingAxes3D
                )
                graspingAxesEndPoints3D = (
                    graspingPositions3D + gipperWidth3D / 2 * graspingAxes3D
                )
                graspingAxesStartPoints2D = eval.reprojectFrom3DRobotBase(
                    graspingAxesStartPoints3D, dataSetPath
                )
                graspingAxesEndPoints2D = eval.reprojectFrom3DRobotBase(
                    graspingAxesEndPoints3D, dataSetPath
                )
                # 2D grasping axes
                graspingAxes2D = graspingAxesEndPoints2D - graspingAxesStartPoints2D
                rgbImg = plotPredictedGraspingPose(
                    rgbImg=rgbImg,
                    graspingPositions2D=graspingPositions2D,
                    graspingAxes2D=graspingAxes2D,
                    predictionColor=styleOpt["predictionColors"][method],
                )
            if controlOpt["showPlot"]:
                eval.plotImageWithMatplotlib(
                    rgbImg, title="grasping prediction", block=True
                )

            if controlOpt["save"]:
                dataSetName = result["dataSetName"]
                fileName = (
                    saveOpt["saveFileNames"]["prediction"]
                    + "_"
                    + str(nRegistrationResult)
                )
                saveFolderPath = saveOpt["saveFolder"]
                saveFolderPath_prediction = os.path.join(
                    saveFolderPath, dataSetName, "predictionResults"
                )
                saveFilePath_prediction = os.path.join(
                    saveFolderPath_prediction, fileName
                )
                if not os.path.exists(saveFolderPath_prediction):
                    os.makedirs(saveFolderPath_prediction, exist_ok=True)
                eval.saveImage(rgbImg, saveFilePath_prediction)
                if controlOpt["verbose"]:
                    print(
                        "Saved prediction {}/{} of result {}/{} at {}".format(
                            nRegistrationResult + 1,
                            len(registrationResultsToEvaluate),
                            nResult + 1,
                            len(resultsToEvaluate),
                            saveFilePath_prediction,
                        )
                    )
                if controlOpt["saveManipulationImg"]:
                    graspImg = eval.getDataSet(frame + 1, dataSetPath)[0]
                    fileName = (
                        saveOpt["saveFileNames"]["manipulation"]
                        + "_"
                        + str(nRegistrationResult)
                    )
                    saveFolderPath_manipulation = os.path.join(
                        saveFolderPath, dataSetName, "graspImgs"
                    )
                    saveFilePath_manipulation = os.path.join(
                        saveFolderPath_manipulation, fileName
                    )
                if not os.path.exists(saveFolderPath_manipulation):
                    os.makedirs(saveFolderPath_manipulation, exist_ok=True)
                eval.saveImage(graspImg, saveFilePath_manipulation)
                if controlOpt["verbose"]:
                    print(
                        "Saved grasp img {}/{} of result {}/{} at {}".format(
                            nRegistrationResult + 1,
                            len(registrationResultsToEvaluate),
                            nResult + 1,
                            len(resultsToEvaluate),
                            saveFilePath_prediction,
                        )
                    )
    if controlOpt["verbose"]:
        print("Finished result generation.")
