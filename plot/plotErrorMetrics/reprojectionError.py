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
    from src.visualization.plotUtils import *
    from src.visualization.plot3D import *
    from src.visualization.plot2D import *
    from src.visualization.colors import *
except:
    print("Imports for plotting error metrics failed.")
    raise

# configs
global eval
eval = InitialLocalizationEvaluation()
runOpt = {"showPlots": True, "savePlots": True}
resultFolderPaths = [
    "data/eval/initialLocalization/results/20230516_112207_YShape",
    "data/eval/initialLocalization/results/20230603_143937_modelY",
    "data/eval/initialLocalization/results/20230516_112207_YShape",
    "data/eval/initialLocalization/results/20230807_150735_partial",
    "data/eval/initialLocalization/results/20230516_113957_Partial",
    "data/eval/initialLocalization/results/20230516_115857_arena",
    "data/eval/initialLocalization/results/20230603_140143_arena",
]
saveFolderPaths = ["imgs/errorMetrics/reprojectionError"]
stlyeConfig = {
    "dataSet": 0,
    "frameConfig_1": 0,
    "frameConfig_2": 2,
    "colorConfig_1": [0, 0, 1],
    "colorConfig_2": [1, 0, 0],
    "colorErrors": [0.2, 0.2, 0.2],
    "elev": 40,
    "azim": 115,
    "dpi": 150,
}
set_text_to_latex_font(scale_axes_labelsize=1.2)


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

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.subplots_adjust(
        left=0.2, bottom=0.1, right=None, top=None, wspace=None, hspace=None
    )
    ax.set_xlabel("$[u] = px$")
    ax.set_ylabel("$[v] = px$")
    ax.imshow(img, cmap="gray", aspect="auto")
    return fig, ax


def plotPointWiseCartesianDistance(resultConfig_1, resultConfig_2):
    fig, ax = setupLatexPlot3D(
        figureWidth=600, xlabel="$[x] = m$", ylabel="$[y] = m$", zlabel="$[z] = m$"
    )
    # Y, _ = eval.getPointCloud(result["frame"], result["dataSetPath"])
    model = eval.generateModel(resultConfig_1["modelParameters"])

    # draw configuration 1
    q = resultConfig_1["localizationResult"]["q"]
    model.setGeneralizedCoordinates(q)
    X_1, adjacencyMatrix = model.getJointPositionsAndAdjacencyMatrix()
    adjacencyMatrix = adjacencyMatrix + adjacencyMatrix.T  # make matrix symmetric
    plotGraph3D(
        ax=ax,
        X=X_1,
        adjacencyMatrix=adjacencyMatrix,
        pointColor=stlyeConfig["colorConfig_1"],
    )

    # fig, ax = setupLatexPlot3D()
    # Y, _ = eval.getPointCloud(result["frame"], result["dataSetPath"])
    model = eval.generateModel(resultConfig_2["modelParameters"])

    # draw configuration 2
    q = resultConfig_2["localizationResult"]["q"]
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
    ax.view_init(elev=stlyeConfig["elev"], azim=stlyeConfig["azim"])
    fig.subplots_adjust(
        left=None, bottom=0.1, right=None, top=None, wspace=None, hspace=None
    )

    # create legend
    markerConfig_1 = configureLegendSymbol(
        style="pointWithLine", color=stlyeConfig["colorConfig_1"]
    )
    markerConfig_2 = configureLegendSymbol(
        style="pointWithLine", color=stlyeConfig["colorConfig_2"]
    )
    markerCorrespondances = configureLegendSymbol(
        style="line", color=stlyeConfig["colorErrors"]
    )
    legendSymbols = [markerConfig_1, markerConfig_2, markerCorrespondances]
    ax.legend(
        handles=legendSymbols,
        labels=["Configuration 1", "Configuration 2", "Pointwise distances"],
        # loc="upper right",
        loc="upper center",
    )
    return fig, ax


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
    fig_2D, ax_2D = plotReprojectionErrors(
        img=backgroundImg, resultConfig_1=resultConfig_1, resultConfig_2=resultConfig_2
    )
    fig_3D, ax_3D = plotPointWiseCartesianDistance(resultConfig_1, resultConfig_2)
    if runOpt["showPlots"]:
        plt.show(block=True)

    if runOpt["savePlots"]:
        # save image
        savePath_2D = os.path.join(saveFolderPaths[0], "reprojectionError2D.pdf")
        fig_2D.savefig(
            savePath_2D, dpi=stlyeConfig["dpi"], format="pdf", bbox_inches="tight"
        )

        savePath_3D = os.path.join(saveFolderPaths[0], "cartesianDistances3D.pdf")
        fig_3D.savefig(
            savePath_3D,
            dpi=stlyeConfig["dpi"],
            format="pdf",
            bbox_inches="tight",
            pad_inches=0.3,
        )
        # eval.saveImage(rgbImage=reprojectionErrorImg, savePath=savePath, verbose=True)
    print("Done.")
