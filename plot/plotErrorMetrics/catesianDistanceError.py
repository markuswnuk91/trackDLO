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

if __name__ == "__main__":
    # load first configuration
    resultFolderPath = resultFolderPaths[stlyeConfig["dataSet"]]
    resultFiles = eval.list_result_files(resultFolderPath)
    result = eval.loadResults(
        os.path.join(resultFolderPath, resultFiles[stlyeConfig["frameConfig_1"]])
    )

    img = eval.getImage(result["frame"], result["dataSetPath"])

    fig, ax = setupLatexPlot3D()
    # Y, _ = eval.getPointCloud(result["frame"], result["dataSetPath"])
    model = eval.generateModel(result["modelParameters"])

    # draw configuration 1
    q = result["localizationResult"]["q"]
    model.setGeneralizedCoordinates(q)
    X_1, adjacencyMatrix = model.getJointPositionsAndAdjacencyMatrix()
    adjacencyMatrix = adjacencyMatrix + adjacencyMatrix.T  # make matrix symmetric
    plotGraph3D(
        ax=ax,
        X=X_1,
        adjacencyMatrix=adjacencyMatrix,
        pointColor=stlyeConfig["colorConfig_1"],
    )
    plt.show(block=False)
    # load second configuration
    result = eval.loadResults(
        os.path.join(resultFolderPath, resultFiles[stlyeConfig["frameConfig_2"]])
    )
    groundTrutPixelCoordinatesConfig_1, _ = eval.loadGroundTruthLabelPixelCoordinates(
        result["filePath"]
    )

    # fig, ax = setupLatexPlot3D()
    # Y, _ = eval.getPointCloud(result["frame"], result["dataSetPath"])
    model = eval.generateModel(result["modelParameters"])

    # draw configuration 2
    q = result["localizationResult"]["q"]
    model.setGeneralizedCoordinates(q)
    X_2, adjacencyMatrix = model.getJointPositionsAndAdjacencyMatrix()
    adjacencyMatrix = adjacencyMatrix + adjacencyMatrix.T  # make matrix symmetric
    plotGraph3D(
        ax=ax,
        X=X_2,
        adjacencyMatrix=adjacencyMatrix,
        pointColor=stlyeConfig["colorConfig_2"],
        lineColor=stlyeConfig["colorConfig_2"],
    )

    plotCorrespondances3D(
        ax=ax,
        X=X_1,
        Y=X_2,
        C=np.eye(len(X_1)),
        xColor=stlyeConfig["colorConfig_1"],
        yColor=stlyeConfig["colorConfig_2"],
        correspondanceColor=stlyeConfig["colorErrors"],
    )
    scale_axes_to_fit(ax=ax, points=X_1, zoom=1)
    plt.show(block=True)
