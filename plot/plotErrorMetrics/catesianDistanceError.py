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
    "data/eval/initialLocalization/results/20230603_143937_modelY",
    "data/eval/initialLocalization/results/20230516_112207_YShape",
    "data/eval/initialLocalization/results/20230807_150735_partial",
    "data/eval/initialLocalization/results/20230516_113957_Partial",
    "data/eval/initialLocalization/results/20230516_115857_arena",
    "data/eval/initialLocalization/results/20230603_140143_arena",
]

if __name__ == "__main__":
    # load first configuration
    resultFolderPath = resultFolderPaths[0]
    resultFiles = eval.list_result_files(resultFolderPath)
    result = eval.loadResults(os.path.join(resultFolderPath, resultFiles[0]))

    img = eval.getImage(result["frame"], result["dataSetPath"])

    fig, ax = setupLatexPlot3D()
    # Y, _ = eval.getPointCloud(result["frame"], result["dataSetPath"])
    model = eval.generateModel(result["modelParameters"])

    # draw configuration 1
    q = result["localizationResult"]["q"]
    model.setGeneralizedCoordinates(q)
    X, adjacencyMatrix = model.getJointPositionsAndAdjacencyMatrix()
    adjacencyMatrix = adjacencyMatrix + adjacencyMatrix.T  # make matrix symmetric
    plotGraph3D(ax=ax, X=X, adjacencyMatrix=adjacencyMatrix)

    # load second configuration
    plt.show(block=True)
