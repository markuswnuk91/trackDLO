import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import time

try:
    sys.path.append(os.getcwd().replace("/eval", ""))
    from src.evaluation.graspingAccuracy.graspingAccuracyEvaluation import (
        GraspingAccuracyEvaluation,
    )
    from src.evaluation.tracking.trackingEvaluation import TrackingEvaluation
    from src.visualization.plot2D import *
    from src.visualization.colors import *
except:
    print("Imports for plotting script tracking error time series failed.")
    raise

global eval
eval = GraspingAccuracyEvaluation()

controlOpt = {
    "resultsToLoad": [8],  # -1
    "registrationResultsToEvaluate": [-1],
    "save": True,
    "verbose": False,
    "showResult_Sim": False, # cannot save while showing
}

saveOpt = {
    "saveFolder": "data/eval/graspingAccuracy/plots/registrationResultsSim3D",
    "saveName": "result",
}

styleOpt = {
    "colorPalette": thesisColorPalettes["viridis"],
    "lineThickness": 5,
    "circleRadius": 10,
    "plotPointCloud": False,
    "pointCloudColor": [1, 0, 0],
    "pointCloudPointSize": 1,
    "targetPointSize": 10,
    "pointCloudAlpha": 1,
    "boardColor": [1, 1, 1],
    "camEye": [0.4, 1.3, 1.3],
    "camCenter": [0.4, 0, 0],
    "camUp": [0, 0, 1],
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

def plotInitializaitonResult_Sim(dataSetResult):
    modelParameters = dataSetResult["initializationResult"]["localizationResult"]["modelParameters"]
    bdloModel = eval.generateModel(modelParameters)

    q = bdloModel.getGeneralizedCoordinates()
    xyzOffset = np.array(([0.425, 0.455, 0.265])) ##compute good intial pose
    rotOffset = [np.pi / 2, -np.pi / 2, 0]
    q[3:6] = xyzOffset
    q[:3] = bdloModel.convertExtrinsicEulerAnglesToBallJointPositions(
        rotOffset[0], rotOffset[1], rotOffset[2]
    )

    bdloModel.setBranchColorsFromColorPalette(styleOpt["colorPalette"])
    trackingEvaluation = TrackingEvaluation()
    dartScene = trackingEvaluation.plotTrackingResultDartSim(
        q=q,
        bdloModel=bdloModel,
        camEye=styleOpt["camEye"],
        camCenter=styleOpt["camCenter"],
        camUp=styleOpt["camUp"],
    )
    dartScene.boardSkel.setColor(styleOpt["boardColor"])
    robotState = eval.loadRobotState(
        dataSetFolderPath=dataSetResult["dataSetPath"],
        fileNumber=0,
    )
    q_robot = robotState["q"] + [0.03, 0.03]
    dartScene.robotSkel.setPositions(q_robot)
    # setup image size
    dataSetPath = dataSetResult["dataSetPath"]
    rgbImg = eval.getDataSet(0, dataSetPath)[0]
    imgShape = rgbImg.shape
    imgWidth = imgShape[1]
    imgHeight = imgShape[0]
    dartScene.setupSceneViewInWindow(x=0, y=0, width=imgWidth, height=imgHeight)

    if styleOpt["plotPointCloud"]:
        pointCloud = dataSetResult["initializationResult"]["localizationResult"]["Y"]
        pointCloudSize = 0.005 if styleOpt["pointCloudPointSize"] is None else styleOpt["pointCloudPointSize"]
        pointCloudColor = [1, 0, 0] if styleOpt["pointCloudColor"] is None else styleOpt["pointCloudColor"]
        pointCloudAlpha = 1 if styleOpt["pointCloudAlpha"] is None else styleOpt["pointCloudAlpha"]
        dartScene.addPointCloud(
            points=pointCloud,
            colors=styleOpt["pointCloudColor"],
            alpha=styleOpt["pointCloudAlpha"],
        )

    if controlOpt["showResult_Sim"]:
        dartScene.viewer.run()

    # save simulation result
    if controlOpt["save"]:
        dataSetPath = dataSetResult["dataSetPath"]
        dataSetName = dataSetPath.split("/")[-2]
        folderPath = os.path.join(saveOpt["saveFolder"], dataSetName, "sim")
        if not os.path.exists(folderPath):
            os.makedirs(folderPath, exist_ok=True)
        saveFilePath = os.path.join(
            saveOpt["saveFolder"],
            dataSetName,
            "sim",
            saveOpt["saveName"]
            + "_sim"
            + "_init"
            + ".png",
        )
        dartScene.viewer.frame()
        time.sleep(0.2)  # sleep to give rendere time to draw
        dartScene.viewer.captureScreen(saveFilePath)
        dartScene.viewer.frame()
        time.sleep(0.2)  # sleep to give rendere time to draw
    return


def plotResult_Sim(dataSetResult, numRegistrationResult, frame_offset=1, filename=None):
    registrationResult = dataSetResult["trackingResults"]["kpr"]["registrationResults"][
        numRegistrationResult
    ]
    pointCloud = registrationResult["result"]["log"]["Y"][0]
    modelParameters = dataSetResult["trackingResults"]["kpr"]["modelParameters"]

    bdloModel = eval.generateModel(modelParameters)
    q = registrationResult["result"]["q"]
    bdloModel.setBranchColorsFromColorPalette(styleOpt["colorPalette"])
    trackingEvaluation = TrackingEvaluation()
    dartScene = trackingEvaluation.plotTrackingResultDartSim(
        q=q,
        bdloModel=bdloModel,
        camEye=styleOpt["camEye"],
        camCenter=styleOpt["camCenter"],
        camUp=styleOpt["camUp"],
    )
    dartScene.boardSkel.setColor(styleOpt["boardColor"])

    frame = registrationResult["frame"]
    frame_grasp = frame + frame_offset
    robotState = eval.loadRobotState(
        dataSetFolderPath=dataSetResult["dataSetPath"],
        fileNumber=frame_grasp,
    )
    q_robot = robotState["q"] + [0.03, 0.03]
    dartScene.robotSkel.setPositions(q_robot)
    # setup image size
    dataSetPath = dataSetResult["dataSetPath"]
    rgbImg = eval.getDataSet(numRegistrationResult, dataSetPath)[0]
    imgShape = rgbImg.shape
    imgWidth = imgShape[1]
    imgHeight = imgShape[0]
    dartScene.setupSceneViewInWindow(x=0, y=0, width=imgWidth, height=imgHeight)

    if styleOpt["plotPointCloud"]:
        pointCloud = registrationResult["Y"]
        pointCloudSize = 0.005 if pointCloudSize is None else pointCloudSize
        pointCloudColor = [1, 0, 0] if pointCloudColor is None else pointCloudColor
        pointCloudAlpha = 1 if pointCloudAlpha is None else pointCloudAlpha
        dartScene.addPointCloud(
            points=pointCloud,
            colors=styleOpt["pointCloudColor"],
            alpha=styleOpt["pointCloudAlpha"],
        )

    if controlOpt["showResult_Sim"]:
        dartScene.viewer.run()

    # save simulation result
    if controlOpt["save"]:
        dataSetPath = dataSetResult["dataSetPath"]
        dataSetName = dataSetPath.split("/")[-2]
        folderPath = os.path.join(saveOpt["saveFolder"], dataSetName, "sim")
        if not os.path.exists(folderPath):
            os.makedirs(folderPath, exist_ok=True)
        if filename is None:
            saveFilePath = os.path.join(
                saveOpt["saveFolder"],
                dataSetName,
                "sim",
                saveOpt["saveName"]
                + "_sim"
                + "_grasp_"
                + str(nRegistrationResult)
                + ".png",
            )
        else:
            saveFilePath = os.path.join(
                saveOpt["saveFolder"], dataSetName, "sim", filename
            )
        dartScene.viewer.frame()
        time.sleep(0.2)  # sleep to give rendere time to draw
        dartScene.viewer.captureScreen(saveFilePath)
        dartScene.viewer.frame()
        time.sleep(0.2)  # sleep to give rendere time to draw
    return

def plotPointCloud_Sim(dataSetResult, numRegistrationResult, frame_offset=1, filename=None):
    registrationResult = dataSetResult["trackingResults"]["kpr"]["registrationResults"][
        numRegistrationResult
    ]
    modelParameters = dataSetResult["trackingResults"]["kpr"]["modelParameters"]

    bdloModel = eval.generateModel(modelParameters)
    q = registrationResult["result"]["q"]
    bdloModel.setBranchColorsFromColorPalette(styleOpt["colorPalette"])

    dataSetIdentifier = dataSetResult["dataSetName"]
    relConfigFilePath = (
            "/evalConfigs/evalConfig" + "_" + dataSetIdentifier + ".json"
        )
    pathToConfigFile = (
            os.path.dirname(os.path.abspath(__file__)) + relConfigFilePath
        )
    trackingEvaluation = TrackingEvaluation(configFilePath=pathToConfigFile)
    dartScene = trackingEvaluation.plotTrackingResultDartSim(
        q=q,
        bdloModel=bdloModel,
        camEye=styleOpt["camEye"],
        camCenter=styleOpt["camCenter"],
        camUp=styleOpt["camUp"],
    )
    dartScene.boardSkel.setColor(styleOpt["boardColor"])

    frame = registrationResult["frame"]
    frame_grasp = frame + frame_offset
    robotState = eval.loadRobotState(
        dataSetFolderPath=dataSetResult["dataSetPath"],
        fileNumber=frame_grasp,
    )
    q_robot = robotState["q"] + [0.03, 0.03]
    dartScene.robotSkel.setPositions(q_robot)
    
    #remove bdlo skel
    dartScene.world.removeSkeleton(dartScene.skel)

    # setup image size
    dataSetPath = dataSetResult["dataSetPath"]
    rgbImg = eval.getDataSet(numRegistrationResult, dataSetPath)[0]
    imgShape = rgbImg.shape
    imgWidth = imgShape[1]
    imgHeight = imgShape[0]
    dartScene.setupSceneViewInWindow(x=0, y=0, width=imgWidth, height=imgHeight)

    #pointCloud = dataSetResult["initializationResult"]["localizationResult"]["Y"]
    pointCloud = trackingEvaluation.getPointCloud(frame_grasp, dataSetResult["dataSetPath"])[0]
    #pointCloud = dataSetResult["trackingResults"]["kpr"]["registrationResults"][numRegistrationResult]["result"]["log"]["Y"][0]
    pointCloudSize = 0.005 if styleOpt["pointCloudPointSize"] is None else styleOpt["pointCloudPointSize"]
    pointCloudColor = [1, 0, 0] if styleOpt["pointCloudColor"] is None else styleOpt["pointCloudColor"]
    pointCloudAlpha = 1 if styleOpt["pointCloudAlpha"] is None else styleOpt["pointCloudAlpha"]
    dartScene.addPointCloud(
        points=pointCloud,
        colors=styleOpt["pointCloudColor"],
        alpha=styleOpt["pointCloudAlpha"],
    )

    if controlOpt["showResult_Sim"]:
        dartScene.viewer.run()

    # save simulation result
    if controlOpt["save"]:
        dataSetPath = dataSetResult["dataSetPath"]
        dataSetName = dataSetPath.split("/")[-2]
        folderPath = os.path.join(saveOpt["saveFolder"], dataSetName, "sim")
        if not os.path.exists(folderPath):
            os.makedirs(folderPath, exist_ok=True)
        if filename is None:
            saveFilePath = os.path.join(
                saveOpt["saveFolder"],
                dataSetName,
                "sim",
                saveOpt["saveName"]
                + "_sim"
                + "_pointCloud_"
                + str(nRegistrationResult)
                + ".png",
            )
        else:
            saveFilePath = os.path.join(
                saveOpt["saveFolder"],
                dataSetName,
                "sim",
                filename +".png",
            )
        dartScene.viewer.frame()
        time.sleep(0.2)  # sleep to give rendere time to draw
        dartScene.viewer.captureScreen(saveFilePath)
        dartScene.viewer.frame()
        time.sleep(0.2)  # sleep to give rendere time to draw
    return


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

        numRegistrationResults = eval.getNumRegistrationResults(result)
        if controlOpt["registrationResultsToEvaluate"][0] == -1:
            registrationResultsToEvaluate = list(range(0, numRegistrationResults))
        else:
            registrationResultsToEvaluate = controlOpt["registrationResultsToEvaluate"]

        colorPalette = styleOpt["colorPalette"]
        lineThickness = styleOpt["lineThickness"]
        circleRadius = styleOpt["circleRadius"]

        # plot initial localization frame
        plotInitializaitonResult_Sim(dataSetResult=result)

        # plot initial frame
        plotResult_Sim(
            dataSetResult=result,
            numRegistrationResult=0,
            frame_offset=0,
            filename=saveOpt["saveName"] + "_sim_grasp_init.png",
        )
        plotPointCloud_Sim(
                dataSetResult=result, numRegistrationResult=0, frame_offset=0,filename="result_sim_pointCloud_init"
            )
        # plot last frame
        plotResult_Sim(
            dataSetResult=result,
            numRegistrationResult=registrationResultsToEvaluate[-1],
            frame_offset=0,
            filename=saveOpt["saveName"] + "_sim_grasp_final.png",
        )
        plotPointCloud_Sim(
                dataSetResult=result, numRegistrationResult=registrationResultsToEvaluate[-1], frame_offset=0,filename="result_sim_pointCloud_final"
            )
        # plot grasp frames
        for nRegistrationResult in registrationResultsToEvaluate[:-1]:
            plotResult_Sim(
                dataSetResult=result, numRegistrationResult=nRegistrationResult
            )
            plotPointCloud_Sim(
                dataSetResult=result, numRegistrationResult=nRegistrationResult
            )
            if controlOpt["verbose"]:
                print(
                    "Saved registration result {}/{} of experiment {}/{} at {}".format(
                        nRegistrationResult + 1,
                        len(registrationResultsToEvaluate),
                        nResult + 1,
                        len(resultsToEvaluate),
                        saveOpt["saveFolder"],
                    )
                )
    if controlOpt["verbose"]:
        print("Finished result generation.")
