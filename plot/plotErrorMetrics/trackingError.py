import sys
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance_matrix

try:
    sys.path.append(os.getcwd().replace("/eval", ""))
    from src.evaluation.initialLocalization.initialLocalizationEvaluation import (
        InitialLocalizationEvaluation,
    )
    from src.localization.bdloLocalization import (
        BDLOLocalization,
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
runOpt = {"showPlots": True, "savePlots": True, "showInitialConfiguration": False}
resultFolderPaths = [
    "data/eval/initialLocalization/results/20230516_112207_YShape",
    "data/eval/initialLocalization/results/20230603_143937_modelY",
    "data/eval/initialLocalization/results/20230516_112207_YShape",
    "data/eval/initialLocalization/results/20230807_150735_partial",
    "data/eval/initialLocalization/results/20230516_113957_Partial",
    "data/eval/initialLocalization/results/20230516_115857_arena",
    "data/eval/initialLocalization/results/20230603_140143_arena",
]
saveFolderPaths = ["imgs/errorMetrics/trackingError"]
styleOpt = {
    "dataSet": 0,
    "figureWidth": 500,
    "initalPositionOffset": np.array([0.1, -0.1, 0]),
    "downsamplingFactor": 9,
    "pointCloudSize": 3,
    "snapshotFactor": 0.3,
    "modelColor": [0, 0, 1],
    "pointCloudColor": [1, 0, 0],
    "pointCloudAlpha": 0.9,
    "correspondanceColor": [0.4, 0.4, 0.4],
    "correspondanceAlpha": 0.1,
    "dpi": 150,
}
set_text_to_latex_font(scale_axes_labelsize=1.2)


def plotTrackingError(model, q, pointCloud, plotLegend=False):
    model.setGeneralizedCoordinates(q)
    X, A_adj = model.getJointPositionsAndAdjacencyMatrix()
    trackingError = (
        1 / len(pointCloud) * np.sum(np.min(distance_matrix(X, pointCloud), axis=0))
    )
    X_2D = X[:, :2]
    pointCloud_2D = pointCloud[:, :2]
    # determine correspondance matrix
    corresponding_indices = np.argmin(distance_matrix(X, pointCloud), axis=0)
    C = np.zeros(shape=(len(X_2D), len(pointCloud_2D)))
    for pcIdx, nodeIdx in enumerate(corresponding_indices):
        C[nodeIdx, pcIdx] = 1
    fig, ax = setupLatexPlot2D(
        figureWidth=styleOpt["figureWidth"], xlabel="$x$ in $m$", ylabel="$y$ in $m$"
    )
    plotGraph2D(ax=ax, X=X_2D, adjacencyMatrix=A_adj, color=styleOpt["modelColor"])
    plotPointSet2D(
        ax=ax,
        X=pointCloud_2D,
        color=styleOpt["pointCloudColor"],
        size=styleOpt["pointCloudSize"],
        alpha=0.3,
    )
    plotMinimalDistances2D(
        ax=ax,
        X=X_2D,
        Y=pointCloud_2D,
        correspondanceMatrix=C,
        correspondanceColor=styleOpt["correspondanceColor"],
        xSize=0,
        ySize=0,
        lineAlpha=0.3,
    )
    ax.set_xlim([-0.1, 0.9])
    ax.set_ylim([-0.51, 0.49])
    set_axes_equal(ax=ax)
    ax.invert_xaxis()
    ax.invert_yaxis()

    if plotLegend:
        # create legend
        markerConfig_1 = configureLegendSymbol(
            style="pointWithLine", color=styleOpt["modelColor"]
        )
        markerConfig_2 = configureLegendSymbol(
            marker=".", style="pointWithoutLine", color=styleOpt["pointCloudColor"]
        )
        markerCorrespondances = configureLegendSymbol(
            style="line", color=styleOpt["correspondanceColor"]
        )
        legendSymbols = [markerConfig_1, markerConfig_2, markerCorrespondances]
        ax.legend(
            handles=legendSymbols,
            labels=[
                "Estimated Configuration",
                "Point cloud",
                "Minimal distances",
            ],
            # loc="upper right",
            loc="upper left",
            bbox_to_anchor=[-0.05, 1.2],
        )
    return fig, ax, trackingError


if __name__ == "__main__":
    # load data set
    resultFolderPath = resultFolderPaths[styleOpt["dataSet"]]
    resultFiles = eval.list_result_files(resultFolderPath)
    result = eval.loadResults(os.path.join(resultFolderPath, resultFiles[0]))
    # load point cloud
    pointCloud = result["l1Result"]["Y"][:: styleOpt["downsamplingFactor"], :]

    # set starting configuration
    model = eval.generateModel(result["modelParameters"])
    model.setInitialPose(initialRotation=np.array([np.pi / 2, 0, np.pi / 2]))
    initialPositionCartesianOffset = (
        model.getCartesianJointPositions()[0, :]
        + np.mean(pointCloud, axis=0)
        - np.mean(model.getCartesianJointPositions(), axis=0)
        + styleOpt["initalPositionOffset"]
    )
    model.setInitialPose(initialPosition=initialPositionCartesianOffset)
    if runOpt["showInitialConfiguration"]:
        fig, ax = setupLatexPlot3D()
        plotPointSet(ax=ax, X=model.getCartesianJointPositions())
        plotPointSet(ax=ax, X=pointCloud, color=[1, 0, 0])
    q0 = model.getGeneralizedCoordinates()
    extractedTopology = result["localizationResult"]["extractedTopology"]
    # perform localization
    localCoordinateSamples = np.linspace(0, 1, 10)
    Y = result["localizationResult"]["Y"]
    localization = BDLOLocalization(
        Y=Y,
        S=localCoordinateSamples,
        templateTopology=model,
        extractedTopology=extractedTopology,
        jacobianDamping=0.3,
        dampingAnnealing=0.9,
        minDamping=0.1,
    )
    # visualizationCallback = eval.getVisualizationCallback(localization)
    # localization.registerCallback(visualizationCallback)

    # run inital localization
    qInit = localization.reconstructShape(
        numIter=3,
        verbose=False,
        method="IK",
    )

    # choose three snapshots from localization result
    q_config_1 = q0
    q_config_2 = localization.qLog[
        int(len(localization.qLog) * styleOpt["snapshotFactor"])
    ]
    q_config_3 = localization.qLog[-1]

    # plot tracking error for snapshots
    fig_config_1, ax_config_1, trackingError_config_1 = plotTrackingError(
        model, q_config_1, pointCloud=pointCloud, plotLegend=True
    )
    fig_config_2, ax_config_2, trackingError_config_2 = plotTrackingError(
        model, q_config_2, pointCloud=pointCloud
    )
    fig_config_3, ax_config_3, trackingError_config_3 = plotTrackingError(
        model, q_config_3, pointCloud=pointCloud
    )

    if runOpt["savePlots"]:
        # save config 1
        savePath_config_1 = os.path.join(
            saveFolderPaths[0], "trackingError_config_1.pdf"
        )
        fig_config_1.savefig(
            savePath_config_1, dpi=styleOpt["dpi"], format="pdf", bbox_inches="tight"
        )

        # save config 2
        trackingErrorString_config_2 = "{:.2f}".format(trackingError_config_2 * 100)
        savePath_config_2 = os.path.join(
            saveFolderPaths[0], "trackingError_config_2.pdf"
        )
        fig_config_2.savefig(
            savePath_config_2, dpi=styleOpt["dpi"], format="pdf", bbox_inches="tight"
        )

        # save config 3
        trackingErrorString_config_3 = "{:.2f}".format(trackingError_config_3 * 100)
        savePath_config_3 = os.path.join(
            saveFolderPaths[0], "trackingError_config_3.pdf"
        )
        fig_config_3.savefig(
            savePath_config_3, dpi=styleOpt["dpi"], format="pdf", bbox_inches="tight"
        )

        # write trackingErrors
        trackingErrorString_config_1 = "{:.2f}".format(trackingError_config_1 * 100)
        trackingErrorString_config_2 = "{:.2f}".format(trackingError_config_2 * 100)
        trackingErrorString_config_3 = "{:.2f}".format(trackingError_config_3 * 100)
        with open(os.path.join(saveFolderPaths[0], "trackingErrors.txt"), "w") as file:
            file.write(
                "Tracking Error of Config 1 [cm]: \n{:.2f}\n\n Tracking Error of Config 2 [cm]: \n{:.2f}\n\n Tracking Error of Config 3 [cm]: \n{:.2f}\n\n".format(
                    trackingError_config_1 * 100,
                    trackingError_config_2 * 100,
                    trackingError_config_3 * 100,
                )
            )
    if runOpt["showPlots"]:
        plt.show(block=True)
    print("Done.")
