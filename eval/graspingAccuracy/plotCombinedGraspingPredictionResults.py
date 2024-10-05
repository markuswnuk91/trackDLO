import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

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
    "saveManipulationImg": False,
    "useManipulationImg": True,
    "showPlot": False,
    "save": True,
    "verbose": True,
}

saveOpt = {
    "saveFolder": "data/eval/graspingAccuracy/plots/combinedGraspingPredictionResult",
    "saveFileNames": {
        "prediction": "predictionResult",
        "manipulation": "grasp",
        "legend": "legend",
    },
}

styleOpt = {
    "groundTruthColor": [0, 1, 0],
    "predictionColors": {
        "cpd": thesisColorPalettes["viridis"].to_rgba(0)[:3],
        "spr": thesisColorPalettes["viridis"].to_rgba(0.5)[:3],
        "kpr": thesisColorPalettes["viridis"].to_rgba(1)[:3],
    },
    "gipperWidth3D": 0.15,
    "fingerWidth2D": 0.5,
    "centerThickness": 12,
    "lineThickness": 8,
    "addLegend": False,  # add a legend to the image
    "saveLegend": True,  # save a legend separately as file
    "legendFontSize": 20,
    "dpi": 150,
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

textwidth_in_pt = 483.6969
figureScaling = 0.45
latexFontSize_in_pt = 14
latexFootNoteFontSize_in_pt = 10
desiredFigureWidth = figureScaling * textwidth_in_pt
desiredFigureHeight = figureScaling * textwidth_in_pt
tex_fonts = {
    #    "pgf.texsystem": "pdflatex",
    # Use LaTeX to write all text
    "text.usetex": True,
    "font.family": "serif",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": latexFootNoteFontSize_in_pt,
    "font.size": latexFootNoteFontSize_in_pt,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": latexFootNoteFontSize_in_pt,
    "xtick.labelsize": latexFootNoteFontSize_in_pt,
    "ytick.labelsize": latexFootNoteFontSize_in_pt,
}
matplotlib.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "text.usetex": True,
        "pgf.rcfonts": False,
    }
)
matplotlib.rcParams.update(tex_fonts)


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


def addLegend(ax, methods):
    patches = []
    labels = []
    for method in methods:
        patches.append(
            Patch(facecolor=styleOpt["predictionColors"][method], edgecolor="black")
        )
        labels.append(method.upper())
    ax.legend(handles=patches, labels=labels)
    return fig, ax


def saveLegend():
    fig = plt.figure(figsize=(1, 2))
    ax = fig.add_subplot()
    patches = []
    labels = []
    for method in controlOpt["methodsToEvaluate"]:
        patches.append(
            Patch(facecolor=styleOpt["predictionColors"][method], edgecolor="black")
        )
        labels.append(method.upper())
    # add ground truth
    patches.append(Patch(facecolor=styleOpt["groundTruthColor"], edgecolor="black"))
    labels.append("ground\ntruth")
    ax.legend(
        handles=patches,
        labels=labels,
        loc="center",
        fontsize=styleOpt["legendFontSize"],
    )
    # Turn off axis so only the legend is saved
    ax.axis("off")

    fileName = saveOpt["saveFileNames"]["legend"]
    saveFolderPath = os.path.join(saveOpt["saveFolder"], "legend")
    saveFilePath = os.path.join(saveFolderPath, fileName)
    if not os.path.exists(saveFolderPath):
        os.makedirs(saveFolderPath, exist_ok=True)
    fig.savefig(
        saveFilePath + ".pdf",
        bbox_inches="tight",
        pad_inches=0.01,
        dpi=styleOpt["dpi"],
    )
    plt.show(block=True)
    return


if __name__ == "__main__":
    if styleOpt["saveLegend"]:
        saveLegend()
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
                    if controlOpt["useManipulationImg"]:
                        image_index = frame + 1
                    else:
                        image_index = frame
                    rgbImg = eval.getDataSet(image_index, dataSetPath)[0]

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
            fig, ax = eval.convertImageToFigure(rgbImg)
            if styleOpt["addLegend"] and nRegistrationResult == 0:
                ax = addLegend(ax, methodsToEvaluate)
            if controlOpt["showPlot"]:
                plt.show(block=True)
                # fig.show(block=True)
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
                fig.savefig(
                    saveFilePath_prediction + ".png",
                    format="png",
                    bbox_inches="tight",
                    dpi=styleOpt["dpi"],
                )
                plt.close("all")
                # eval.saveImage(rgbImg, saveFilePath_prediction)
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
                    # eval.saveImage(graspImg, saveFilePath_manipulation)
                    fig, ax = eval.convertImageToFigure(graspImg)
                    fig.savefig(
                        saveFilePath_manipulation + ".png",
                        format="png",
                        bbox_inches="tight",
                        dpi=styleOpt["dpi"],
                    )
                    plt.close("all")
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
