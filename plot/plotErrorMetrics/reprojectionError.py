import sys
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

try:
    sys.path.append(os.getcwd().replace("/eval", ""))
    from src.evaluation.initialLocalization.initialLocalizationEvaluation import (
        InitialLocalizationEvaluation,
    )
    from src.visualization.plot3D import *
    from src.visualization.plot2D import *
    from src.visualization.colors import *
except:
    print("Imports for plotting error metrics failed.")
    raise

# configs
global eval
eval = InitialLocalizationEvaluation()
resultFolderPaths = [
    "data/eval/initialLocalization/results/20230516_112207_YShape",
    "data/eval/initialLocalization/results/20230603_143937_modelY",
    "data/eval/initialLocalization/results/20230516_112207_YShape",
    "data/eval/initialLocalization/results/20230807_150735_partial",
    "data/eval/initialLocalization/results/20230516_113957_Partial",
    "data/eval/initialLocalization/results/20230516_115857_arena",
    "data/eval/initialLocalization/results/20230603_140143_arena",
]
stlyeConfig = {
    "dataSet": 0,
    "frameConfig_1": 0,
    "frameConfig_2": 2,
    "colorConfig_1": [0, 0, 1],
    "colorConfig_2": [1, 0, 0],
    "colorErrors": [0.2, 0.2, 0.2],
}


def plotReprojectionErrors(
    img,
    resultConfig_1,
    resultConfig_2,
):
    # plot first configuration
    referencePositionsConfig_1 = eval.extractReferencePositions(resultConfig_1)
    adjacencyMatrixConfig_1 = referencePositionsConfig_1["adjacencyMatrix"]
    positions2DConfig_1 = referencePositionsConfig_1["jointCoordinates2D"]
    img = plotGraph2D(
        rgbImg=img,
        positions2D=positions2DConfig_1,
        adjacencyMatrix=adjacencyMatrixConfig_1,
        lineColor=stlyeConfig["colorConfig_1"],
        circleColor=stlyeConfig["colorConfig_1"],
        lineThickness=5,
        circleRadius=1,
    )

    # plot second configuration
    referencePositionsConfig_2 = eval.extractReferencePositions(resultConfig_2)
    adjacencyMatrixConfig_2 = referencePositionsConfig_2["adjacencyMatrix"]
    positions2DConfig_2 = referencePositionsConfig_2["jointCoordinates2D"]
    img = plotGraph2D(
        rgbImg=img,
        positions2D=positions2DConfig_2,
        adjacencyMatrix=adjacencyMatrixConfig_2,
        lineColor=stlyeConfig["colorConfig_2"],
        circleColor=stlyeConfig["colorConfig_2"],
        lineThickness=5,
        circleRadius=1,
    )
    # plot corresondances
    img = plotCorrespondances2D(
        rgbImg=img,
        predictionPixelCoordinates=positions2DConfig_1,
        groundTruthPixelCoordinates=positions2DConfig_2,
        predictionColor=stlyeConfig["colorConfig_1"],
        groundTruthColor=stlyeConfig["colorConfig_2"],
        correspondanceColor=stlyeConfig["colorErrors"],
        correspondanceLineWidth=5,
        predictionCircleRadius=10,
        groundTruthCircleRadius=10,
    )
    return img


if __name__ == "__main__":
    # load first configuration result
    resultFolderPath = resultFolderPaths[stlyeConfig["dataSet"]]
    resultFiles = eval.list_result_files(resultFolderPath)
    resultConfig_1 = eval.loadResults(
        os.path.join(resultFolderPath, resultFiles[stlyeConfig["frameConfig_1"]])
    )
    # load second configuration result
    resultConfig_2 = eval.loadResults(
        os.path.join(resultFolderPath, resultFiles[stlyeConfig["frameConfig_2"]])
    )
    # groundTrutPixelCoordinatesConfig_1, _ = eval.loadGroundTruthLabelPixelCoordinates(
    #     result["filePath"]
    # )
    backgroundImg = eval.getImage(
        resultConfig_2["frame"], resultConfig_2["dataSetPath"]
    )
    reprojectionErrorImg = plotReprojectionErrors(
        img=backgroundImg, resultConfig_1=resultConfig_1, resultConfig_2=resultConfig_2
    )
    eval.plotImageWithMatplotlib(rgbImage=reprojectionErrorImg, block=True)
