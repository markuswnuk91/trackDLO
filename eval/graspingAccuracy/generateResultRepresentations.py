import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from warnings import warn

try:
    sys.path.append(os.getcwd().replace("/eval", ""))
    from src.evaluation.graspingAccuracy.graspingAccuracyEvaluation import (
        GraspingAccuracyEvaluation,
    )
except:
    print("Imports for Grasping Accuracy Result Evaluation failed.")
    raise
global eval
pathToConfigFile = (
    os.path.dirname(os.path.abspath(__file__))
    + "/evalConfigs/representationConfig.json"
)
eval = GraspingAccuracyEvaluation(configFilePath=pathToConfigFile)

resultRootFolderPath = eval.config["resultRootFolderPath"]
resultFileName = eval.config["resultFileName"]


def loadResults(resultRootFolderPath):
    results = []
    subfolders = [
        f
        for f in os.listdir(resultRootFolderPath)
        if os.path.isdir(os.path.join(resultRootFolderPath, f))
    ]
    for subfolder in subfolders:
        resultFolderPath = os.path.join(resultRootFolderPath, subfolder)
        resultFileNames = [
            f
            for f in os.listdir(resultFolderPath)
            if os.path.isfile(os.path.join(resultFolderPath, f)) and resultFileName in f
        ]
        if len(resultFileNames) > 1:
            warn(
                "Multiple result files found in folder {}. Using only the first one.".format(
                    resultFolderPath
                )
            )
        resultFilePath = os.path.join(resultFolderPath, resultFileNames[0])
        result = eval.loadResults(resultFilePath)
        results.append(result)

    return results


def accumulateGraspingErrors(results):
    translationalGraspingErrors = []
    rotationalGraspingErrors = []
    correspondingMethods = []
    correspondingModelNames = []
    for dataSetIndex, result in enumerate(results):
        for registrationMethodIndex, registrationMethodResult in enumerate(
            result["graspingAccuracyResults"]
        ):
            registrationMethod = registrationMethodResult["method"]
            numGraspingPositions = len(
                registrationMethodResult["graspingPositionErrors"]
            )
            for graspingIndex in range(0, numGraspingPositions):
                # get translational grasping error
                translationalGraspingError = results[dataSetIndex][
                    "graspingAccuracyResults"
                ][registrationMethodIndex]["graspingPositionErrors"][graspingIndex]
                translationalGraspingErrors.append(translationalGraspingError)

                if translationalGraspingError > 1.5:
                    print("Here")
                # get rotational grasping error
                rotationalGraspingError = results[dataSetIndex][
                    "graspingAccuracyResults"
                ][registrationMethodIndex]["graspingAngularErrors"]["rad"][
                    graspingIndex
                ]
                rotationalGraspingErrors.append(rotationalGraspingError)

                # get model type
                modelName = results[dataSetIndex]["graspingAccuracyResults"][
                    registrationMethodIndex
                ]["trackingResult"]["initializationResult"]["modelParameters"][
                    "modelInfo"
                ][
                    "name"
                ]
                correspondingModelNames.append(modelName)

                # get corresponding method
                correspondingMethods.append(registrationMethod)
    return (
        translationalGraspingErrors,
        rotationalGraspingErrors,
        correspondingMethods,
        correspondingModelNames,
    )


def scatterPlotGraspingErrors(results):
    alpha = 0.3
    (
        translationalGraspingErrors,
        rotationalGraspingErrors,
        correspondingMethods,
        correspondingModelNames,
    ) = accumulateGraspingErrors(results)

    colors = []
    for method in correspondingMethods:
        if method == "cpd":
            colors.append([1, 0, 0, alpha])
        elif method == "spr":
            colors.append([0, 0, 1, alpha])
        elif method == "krcpd":
            colors.append([0, 1, 0, alpha])
        elif method == "krcpd4BDLO":
            colors.append([1, 1, 0, alpha])
        else:
            colors.append([0.7, 0.7, 0.7, alpha])
    # for model in correspondingModelNames:
    #     if model == "modelY":
    #         colors.append([1, 0, 0, alpha])
    #     elif model == "partialWireHarness":
    #         colors.append([0, 0, 1, alpha])
    #     elif model == "arenaWireHarness":
    #         colors.append([0, 1, 0, alpha])
    #     elif model == "singleDLO":
    #         colors.append([1, 1, 0, alpha])
    #     else:
    #         colors.append([0.7, 0.7, 0.7, 0.1])

    markers = []
    # for method in correspondingMethods:
    #     if method == "cpd":
    #         markers.append("^")
    #     elif method == "spr":
    #         markers.append("s")
    #     elif method == "krcpd":
    #         markers.append("o")
    #     elif method == "krcpd4BDLO":
    #         markers.append("D")
    #     else:
    #         markers.append(".")
    for model in correspondingModelNames:
        if model == "modelY":
            markers.append("o")
        elif model == "partialWireHarness":
            markers.append("^")
        elif model == "arenaWireHarness":
            markers.append("s")
        elif model == "singleDLO":
            markers.append("D")
        else:
            colors.append([0.7, 0.7, 0.7, 0.1])
    for i, marker in enumerate(markers):
        plt.scatter(
            rotationalGraspingErrors[i],
            translationalGraspingErrors[i],
            c=colors[i],
            marker=marker,
        )
    plt.show(block=True)
    return


def visualizeGraspingError2D(graspingAccuracyResult):
    # parameters
    gripperWidth = 0.1
    fingerWidth = 0.3
    frames = graspingAccuracyResult["predictedFrames"]
    dataSetPath = graspingAccuracyResult["dataSetPath"]
    rgbImages = []
    for frame in frames:
        # load 2D images
        fileName = eval.getFileName(frame, dataSetPath)
        rgbImage, _ = eval.getDataSet(frame, dataSetPath)
        rgbImages.append(rgbImage)

    # reproject ground truth positions in image
    groundTruthGraspingPositions3D = np.array(
        graspingAccuracyResult["graspingPositions"]["groundTruth"]
    )
    groundTruthGraspingPositions2D = eval.reprojectFrom3DRobotBase(
        groundTruthGraspingPositions3D, dataSetPath
    )
    # reporject predictd grasping pose in 2D image
    predictedGraspingPositions3D = np.array(
        graspingAccuracyResult["graspingPositions"]["predicted"]
    )
    predictedGraspingPositions2D = eval.reprojectFrom3DRobotBase(
        predictedGraspingPositions3D, dataSetPath
    )

    # reproject grasping axis
    groundTruthGraspingAxes3D = np.array(
        graspingAccuracyResult["graspingAxes"]["groundTruth"]
    )
    predictedGraspingAxes3D = np.array(
        graspingAccuracyResult["graspingAxes"]["predicted"]
    )
    groundTruthGraspingAxesStartPoints3D = (
        groundTruthGraspingPositions3D - gripperWidth / 2 * groundTruthGraspingAxes3D
    )
    groundTruthGraspingAxesEndPoints3D = (
        groundTruthGraspingPositions3D + gripperWidth / 2 * groundTruthGraspingAxes3D
    )
    predictedGraspingAxesStartPoints3D = (
        predictedGraspingPositions3D - gripperWidth / 2 * predictedGraspingAxes3D
    )
    predictedGraspingAxesEndPoints3D = (
        predictedGraspingPositions3D + gripperWidth / 2 * predictedGraspingAxes3D
    )

    groundTruthGraspingAxesStartPoints2D = eval.reprojectFrom3DRobotBase(
        groundTruthGraspingAxesStartPoints3D, dataSetPath
    )
    groundTruthGraspingAxesEndPoints2D = eval.reprojectFrom3DRobotBase(
        groundTruthGraspingAxesEndPoints3D, dataSetPath
    )

    predictedGraspingAxesEndPoints2D = eval.reprojectFrom3DRobotBase(
        predictedGraspingAxesEndPoints3D, dataSetPath
    )
    predictedGraspingAxesStartPoints2D = eval.reprojectFrom3DRobotBase(
        predictedGraspingAxesStartPoints3D, dataSetPath
    )

    # 2D grasping axes
    groundTruthGraspingAxes2D = (
        groundTruthGraspingAxesEndPoints2D - groundTruthGraspingAxesStartPoints2D
    )
    predictedGraspingAxes2D = (
        predictedGraspingAxesEndPoints2D - predictedGraspingAxesStartPoints2D
    )
    # orthogonal 2D gripper axis
    groundTruthGripperAxes2D = (
        np.array(([0, 1], [-1, 0])) @ groundTruthGraspingAxes2D.T
    ).T
    groundTruthGripperStartPoints2D = np.around(
        (groundTruthGraspingPositions2D - 0.5 * groundTruthGripperAxes2D)
    ).astype(int)
    groundTruthGripperEndPoints2D = np.around(
        groundTruthGraspingPositions2D + 0.5 * groundTruthGripperAxes2D
    ).astype(int)
    predictedGripperAxes2D = (np.array(([0, 1], [-1, 0])) @ predictedGraspingAxes2D.T).T
    predictedGripperStartPoints2D = np.around(
        (predictedGraspingPositions2D - 0.5 * predictedGripperAxes2D)
    ).astype(int)
    predictedGripperEndPoints2D = np.around(
        predictedGraspingPositions2D + 0.5 * predictedGripperAxes2D
    ).astype(int)
    # gripper fingers
    groundTruthGripperEndFingerStartPoints = np.around(
        groundTruthGripperEndPoints2D - 0.5 * fingerWidth * groundTruthGraspingAxes2D
    ).astype(int)
    groundTruthGripperEndFingerEndPoints = np.around(
        groundTruthGripperEndPoints2D + 0.5 * fingerWidth * groundTruthGraspingAxes2D
    ).astype(int)
    groundTruthGripperStartFingerStartPoints = np.around(
        groundTruthGripperStartPoints2D - 0.5 * fingerWidth * groundTruthGraspingAxes2D
    ).astype(int)
    groundTruthGripperStartFingerEndPoints = np.around(
        groundTruthGripperStartPoints2D + 0.5 * fingerWidth * groundTruthGraspingAxes2D
    ).astype(int)
    predictedGripperEndFingerStartPoints = np.around(
        predictedGripperEndPoints2D - 0.5 * fingerWidth * predictedGraspingAxes2D
    ).astype(int)
    predictedGripperEndFingerEndPoints = np.around(
        predictedGripperEndPoints2D + 0.5 * fingerWidth * predictedGraspingAxes2D
    ).astype(int)
    predictedGripperStartFingerStartPoints = np.around(
        predictedGripperStartPoints2D - 0.5 * fingerWidth * predictedGraspingAxes2D
    ).astype(int)
    predictedGripperStartFingerEndPoints = np.around(
        predictedGripperStartPoints2D + 0.5 * fingerWidth * predictedGraspingAxes2D
    ).astype(int)
    # plot everything
    import cv2

    markerThickness = 10
    groundTruthColor = [0, 255, 0]
    predictionColor = [0, 0, 255]
    markerFill = -1

    for i, graspingPosition in enumerate(groundTruthGraspingPositions2D):
        rgbImages[i] = cv2.circle(
            rgbImages[i],
            graspingPosition,
            markerThickness,
            groundTruthColor,
            markerFill,
        )
        rgbImages[i] = cv2.circle(
            rgbImages[i],
            predictedGraspingPositions2D[i],
            markerThickness,
            predictionColor,
            markerFill,
        )
        # rgbImages[i] = cv2.line(
        #     rgbImages[i],
        #     (
        #         groundTruthGraspingAxesStartPoints2D[i][0],
        #         groundTruthGraspingAxesStartPoints2D[i][1],
        #     ),
        #     (
        #         groundTruthGraspingAxesEndPoints2D[i][0],
        #         groundTruthGraspingAxesEndPoints2D[i][1],
        #     ),
        #     groundTruthColor,
        #     5,
        # )
        # rgbImages[i] = cv2.line(
        #     rgbImages[i],
        #     (
        #         predictedGraspingAxesStartPoints2D[i][0],
        #         predictedGraspingAxesStartPoints2D[i][1],
        #     ),
        #     (
        #         predictedGraspingAxesEndPoints2D[i][0],
        #         predictedGraspingAxesEndPoints2D[i][1],
        #     ),
        #     predictionColor,
        #     5,
        # )

        # ground truth
        # gripper axis
        rgbImages[i] = cv2.line(
            rgbImages[i],
            (
                groundTruthGripperStartPoints2D[i][0],
                groundTruthGripperStartPoints2D[i][1],
            ),
            (groundTruthGripperEndPoints2D[i][0], groundTruthGripperEndPoints2D[i][1]),
            groundTruthColor,
            5,
        )
        # finger at end
        rgbImages[i] = cv2.line(
            rgbImages[i],
            (
                groundTruthGripperEndFingerStartPoints[i][0],
                groundTruthGripperEndFingerStartPoints[i][1],
            ),
            (
                groundTruthGripperEndFingerEndPoints[i][0],
                groundTruthGripperEndFingerEndPoints[i][1],
            ),
            groundTruthColor,
            5,
        )
        # finger at start
        rgbImages[i] = cv2.line(
            rgbImages[i],
            (
                groundTruthGripperStartFingerStartPoints[i][0],
                groundTruthGripperStartFingerStartPoints[i][1],
            ),
            (
                groundTruthGripperStartFingerEndPoints[i][0],
                groundTruthGripperStartFingerEndPoints[i][1],
            ),
            groundTruthColor,
            5,
        )

        # predicted
        rgbImages[i] = cv2.line(
            rgbImages[i],
            (predictedGripperStartPoints2D[i][0], predictedGripperStartPoints2D[i][1]),
            (predictedGripperEndPoints2D[i][0], predictedGripperEndPoints2D[i][1]),
            predictionColor,
            5,
        )
        # finger at end
        rgbImages[i] = cv2.line(
            rgbImages[i],
            (
                predictedGripperEndFingerStartPoints[i][0],
                predictedGripperEndFingerStartPoints[i][1],
            ),
            (
                predictedGripperEndFingerEndPoints[i][0],
                predictedGripperEndFingerEndPoints[i][1],
            ),
            predictionColor,
            5,
        )
        # finger at start
        rgbImages[i] = cv2.line(
            rgbImages[i],
            (
                predictedGripperStartFingerStartPoints[i][0],
                predictedGripperStartFingerStartPoints[i][1],
            ),
            (
                predictedGripperStartFingerEndPoints[i][0],
                predictedGripperStartFingerEndPoints[i][1],
            ),
            predictionColor,
            5,
        )
    for rgbImage in rgbImages:
        fig = plt.figure(frameon=False)
        # fig.set_size_inches(imageWitdthInInches, imageHeightInInches)
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(rgbImage, aspect="auto")
    plt.show(block=True)
    return


def tabularizeResults(results):
    (
        translationalGraspingErrors,
        rotationalGraspingErrors,
        correspondingMethods,
        correspondingModels,
    ) = accumulateGraspingErrors(results)

    evaluationResults = {
        "translationalError": {},
        "translationalStdDev": {},
        "rotationalError": {},
        "rotationalStdDev": {},
    }
    models = list(set(correspondingModels))
    methods = list(set(correspondingMethods))

    for i, model in enumerate(models):
        for j, method in enumerate(methods):
            # gather all errors for the combination of model and method
            translationalIndices = [
                index
                for index, value in enumerate(translationalGraspingErrors)
                if correspondingModels[index] == model
                and correspondingMethods[index] == method
            ]
            evaluationResults["translationalError"][model] = {}
            evaluationResults["translationalError"][model][method] = np.mean(
                np.array(translationalGraspingErrors)[translationalIndices]
            )
            evaluationResults["translationalStdDev"][model] = {}
            evaluationResults["translationalStdDev"][model][method] = np.std(
                np.array(translationalGraspingErrors)[translationalIndices]
            )

            rotationalIndices = [
                index
                for index, value in enumerate(rotationalGraspingErrors)
                if correspondingModels[index] == model
                and correspondingMethods[index] == method
            ]
            evaluationResults["rotationalError"][model] = {}
            evaluationResults["rotationalError"][model][method] = np.mean(
                np.array(rotationalGraspingErrors)[rotationalIndices]
            )
            evaluationResults["rotationalStdDev"][model] = {}
            evaluationResults["rotationalStdDev"][model][method] = np.std(
                np.array(rotationalGraspingErrors)[rotationalIndices]
            )

    #     f"""
    #     \begin{tabular}{lccccccc}\toprule
    # 	& & \multicolumn{2}{c}{\acs{CPD}} & \multicolumn{2}{c}{\acs{SPR}} & \multicolumn{2}{c}{\acs{KPR}}
    # 	\\\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}
    # 	model		& $n_{\text{grasp}}$  & $\bar{e}_{t}$ & $\sigma_{e_t}$ & $\bar{e}_{t}$ & $\sigma_{e_t}$ & $\bar{e}_{t}$ & $\sigma_{e_t}$ \\\midrule
    # 	\acs{TCL}-0  & x & x & x & x & x& x& x  \\
    # 	\acs{TCL}-1. & x & x & x & x & x& x& x  \\
    # 	\acs{TCL}-2 	& x & x & x & x & x& x& x  \\
    # 	\acs{TCL}-3  & x & x & x & x & x& x& x \\\bottomrule
    # \end{tabular}
    #     latex_table_column = f"""
    #             ${translationalErrorResults['model']}$ & ${translationalErrorResults["model"]['n_config']}$ & ${runtimeResults['t_preprocessing']:.2f}$ & ${runtimeResults['t_l1_skel']:.2f}$ & ${runtimeResults['t_som']:.2f}$ & ${runtimeResults['t_mst']:.2f}$ & ${runtimeResults['t_corresp']:.2f}$ & ${runtimeResults['t_ik']:.2f}$ & $\\mathbf{{{runtimeResults['t_total']:.2f}}}$ \\\\
    #             """"

    # translationalErrorResult["model"]["method"]

    # meanRotationalError["model"]["method"]
    #         translationalErrorResult = {
    #         "model":
    #     }

    # for i, translationalGraspingError in enumerate(translationalGraspingErrors):
    #     translationalErrorResult = {
    #         "model":
    #     }
    # translationalErrorResults = [
    #     {

    #     }
    # ]

    # {
    #     "model": eval.getModelID(
    #         evaluationResults[0]["initializationResult"]["modelParameters"][
    #             "modelInfo"
    #         ]["name"]
    #     ),
    #     "n_grasps": sdf,
    #     "t_preprocessing": 0.0,
    #     "t_l1_skel": 0.0,
    #     "t_som": 0.0,
    #     "t_mst": 0.0,
    #     "t_corresp": 0.0,
    #     "t_ik": 0.0,
    #     "t_total": 0.0,
    # }

    # latex_table_column = f"""
    #         ${translationalErrorResults['model']}$ & ${translationalErrorResults["model"]['n_config']}$ & ${runtimeResults['t_preprocessing']:.2f}$ & ${runtimeResults['t_l1_skel']:.2f}$ & ${runtimeResults['t_som']:.2f}$ & ${runtimeResults['t_mst']:.2f}$ & ${runtimeResults['t_corresp']:.2f}$ & ${runtimeResults['t_ik']:.2f}$ & $\\mathbf{{{runtimeResults['t_total']:.2f}}}$ \\\\
    #         """"

    raise NotImplementedError


if __name__ == "__main__":
    # load all results
    results = loadResults(resultRootFolderPath)

    tabularizeResults(results)
    visualizeGraspingError2D(
        graspingAccuracyResult=results[0]["graspingAccuracyResults"][0]
    )

    # plot result representations
    scatterPlotGraspingErrors(results)
    # tabularize results

    # save result representations

    print("Done.")
