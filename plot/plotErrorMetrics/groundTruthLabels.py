import sys
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

try:
    sys.path.append(os.getcwd().replace("/eval", ""))
    from src.evaluation.graspingAccuracy.graspingAccuracyEvaluation import (
        GraspingAccuracyEvaluation,
    )
    from src.evaluation.initialLocalization.initialLocalizationEvaluation import (
        InitialLocalizationEvaluation,
    )
    from src.visualization.plot3D import *
    from src.visualization.plot2D import *
    from src.visualization.colors import *
except:
    print("Imports for plotting ground labels failed.")
    raise


runOpt = {
    "showPlot": True,
    "savePlots": True,
    "plotManualGroundTruthLabels": True,
    "plotGraspingPoseGroundTruthLabels": True,
    "plotRegistrationResult": False,
}
saveOpt = {
    "savePathLabelImage": "imgs/groundTruthLabels",
    "savePathGraspsImage": "imgs/groundTruthLabels",
}
styleOpt = {
    "groundTruthLabelColor": thesisColors["green"],
    "groundTruthLabelCircleColor": thesisColors["green"],
    "groundTruthLabelCircleRadius": 27,
    "groundTruthLabelCircleThickness": 7,
    "groundTruthLabelCircleInnerPointSize": 5,
    "graspingFrameToLoad": 1,
    "groundTruthGraspingPoseColor": thesisColors["green"],
    "gipperWidth3D": 0.1,
    "fingerWidth2D": 0.5,
    "gripperCenterSize": 10,
    "gripperLineThickness": 7,
}

# configs
global graspingEval
global initEval
graspingEval = GraspingAccuracyEvaluation()
initEval = InitialLocalizationEvaluation()
groundTruthExamplePath_labels = (
    "data/eval/initialLocalization/results/20230516_115857_arena"
)

groundTruthExamplePath_grasps = (
    "data/eval/graspingAccuracy/results/20230522_140014_arena"
)


def getGroundTruthGraspingPoses(
    graspingEvaluationHandle,
    graspingAccuracyEvaluationResult,
    methodForPrediction="kpr",
    frameToLoad=0,
):
    eval = graspingEvaluationHandle
    result = graspingAccuracyEvaluationResult
    method = methodForPrediction
    num = frameToLoad
    registrationResult = result["trackingResults"][method]["registrationResults"][num]
    frame = registrationResult["frame"]
    dataSetPath = result["dataSetPath"]

    # prediction
    graspingLocalCoordinates = eval.loadGraspingLocalCoordinates(dataSetPath)
    predictedGraspingPositions = []
    predictedGraspingAxes = []
    for graspingLocalCoordinate in graspingLocalCoordinates:
        T = registrationResult["result"]["T"]
        B = result["trackingResults"][method]["B"]
        S = result["initializationResult"]["localizationResult"]["SInit"]
        (
            predictedGraspingPosition,
            predictedGraspingAxis,
        ) = eval.predictGraspingPositionAndAxisFromRegistrationTargets(
            T, B, S, graspingLocalCoordinate
        )
        predictedGraspingPositions.append(predictedGraspingPosition)
        predictedGraspingAxes.append(predictedGraspingAxis)

    graspingPositions3D = np.stack(predictedGraspingPositions)
    graspingAxes3D = np.stack(predictedGraspingAxes)
    rgbImg = eval.getDataSet(frame, dataSetPath)[0]
    positions2D = eval.reprojectFrom3DRobotBase(T, dataSetFolderPath=dataSetPath)
    adjacencyMatrix = result["trackingResults"][method]["adjacencyMatrix"]
    if runOpt["plotRegistrationResult"]:
        rgbImg = eval.plotBranchWiseColoredRegistrationResult(
            rgbImg, positions2D, adjacencyMatrix, B
        )
        eval.plotImageWithMatplotlib(rgbImg)
    # reproject grasping positions in image
    graspingPositions2D = eval.reprojectFrom3DRobotBase(
        graspingPositions3D, dataSetPath
    )
    # reproject grasping axes in image
    gipperWidth3D = styleOpt["gipperWidth3D"]
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

    return graspingPositions2D, graspingAxes2D


if __name__ == "__main__":
    if runOpt["plotManualGroundTruthLabels"]:
        # load label example
        resultFileNames = initEval.list_result_files(groundTruthExamplePath_labels)
        frameToLoad = 0
        result_labels = initEval.loadResults(
            os.path.join(groundTruthExamplePath_labels, resultFileNames[frameToLoad])
        )
        imageFilePath = result_labels["filePath"]
        # plot cirles around ground truth circles (wire harness with ground truth labels)
        img_gtLabels = initEval.getImage(
            result_labels["frame"], result_labels["dataSetPath"]
        )
        gtPixelCoordinates = initEval.loadGroundTruthLabelPixelCoordinates(
            imageFilePath
        )[0]
        img_gtLabels = plotCircles_CV(
            rgbImg=img_gtLabels,
            centerCoordinates=gtPixelCoordinates,
            circleColor=styleOpt["groundTruthLabelCircleColor"],
            circleRadius=styleOpt["groundTruthLabelCircleRadius"],
            circleLineWidth=styleOpt["groundTruthLabelCircleThickness"],
        )
        img_gtLabels = plotCircles_CV(
            rgbImg=img_gtLabels,
            centerCoordinates=gtPixelCoordinates,
            circleColor=styleOpt["groundTruthLabelColor"],
            circleRadius=styleOpt["groundTruthLabelCircleInnerPointSize"],
            fill=True,
        )

        # save image
        if runOpt["savePlots"]:
            savePath = os.path.join(saveOpt["savePathLabelImage"], "groundTruthLabels")
            initEval.saveImage(img_gtLabels, savePath, verbose=True)

        if runOpt["showPlot"]:
            initEval.plotImageWithMatplotlib(img_gtLabels)
    if runOpt["plotGraspingPoseGroundTruthLabels"]:
        # load grasping ground truth example (wire harness grasped by manipulator)
        # resultFolderPath = resultFolderPaths[stlyeConfig["dataSet"]]
        result_graspingPose = graspingEval.loadResults(
            os.path.join(groundTruthExamplePath_grasps, "result.pkl")
        )
        dataSetPath = result_graspingPose["dataSetPath"]
        img_gtGrasps = graspingEval.getImage(
            styleOpt["graspingFrameToLoad"], result_graspingPose["dataSetPath"]
        )
        # showImage_CV(img_gtGrasps)
        # get the pixel positions for every ground truth grasping pose
        graspingPositions2D, graspingAxes2D = getGroundTruthGraspingPoses(
            graspingEvaluationHandle=graspingEval,
            graspingAccuracyEvaluationResult=result_graspingPose,
            methodForPrediction="kpr",
            frameToLoad=styleOpt["graspingFrameToLoad"] - 1,
        )

        # plot grasping poses
        for graspingPosition, graspingAxis in zip(graspingPositions2D, graspingAxes2D):
            img_gtGrasps = plotGraspingPose2D(
                rgbImage=img_gtGrasps,
                graspingPosition2D=graspingPosition,
                graspingAxis2D=graspingAxis,
                color=styleOpt["groundTruthGraspingPoseColor"],
                fingerWidth2D=styleOpt["fingerWidth2D"],
                centerThickness=styleOpt["gripperCenterSize"],
                lineThickness=styleOpt["gripperLineThickness"],
                markerFill=-1,
            )
        # save image
        if runOpt["savePlots"]:
            savePath = os.path.join(
                saveOpt["savePathGraspsImage"], "groundTruthGraspingPoses"
            )
            graspingEval.saveImage(img_gtGrasps, savePath, verbose=True)
        if runOpt["showPlot"]:
            graspingEval.plotImageWithMatplotlib(img_gtGrasps)
    if runOpt["showPlot"]:
        plt.show(block=True)
    print("Done.")
