import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib
import traceback

try:
    sys.path.append(os.getcwd().replace("/eval", ""))
    from src.evaluation.initialLocalization.initialLocalizationEvaluation import (
        InitialLocalizationEvaluation,
    )
    from src.visualization.plot3D import *
    from src.visualization.colors import *
except:
    print("Imports for plotting localization results 2D failed.")
    raise

global eval
eval = InitialLocalizationEvaluation()

controlOpt = {
    "dataSetsToLoad": [-1],
    "resultsToLoad": [-1],
    "save": True,
    "showPlot": False,
    "block": False,
    "saveFolder": "data/eval/initialLocalization/plots/inputData",
    "saveNameImg": "inputImg",
    "saveNameInputPointCloud": "inputPointCloud",
    "saveNameSegmentedPointCloud": "segmentedPointCloud",
    "saveAsPGF": False,
    "verbose": True,
    "plotInputPointCloud": True,
    "plotSegmentedPointCloud": True,
}

resultFolderPaths = [
    "data/eval/initialLocalization/results/20230603_143937_modelY",
    "data/eval/initialLocalization/results/20230516_112207_YShape",
    "data/eval/initialLocalization/results/20230807_150735_partial",
    "data/eval/initialLocalization/results/20230516_113957_Partial",
    "data/eval/initialLocalization/results/20230516_115857_arena",
    "data/eval/initialLocalization/results/20230603_140143_arena",
]

styleOpt = {
    "subplot_position_left": 0,
    "subplot_position_bottom": 0,
    "subplot_position_right": 1,
    "subplot_position_top": 1,
    "zoomFactor": 1.5,
    "azimuth": 91,
    "elevation": 50,
    "pointCloudSize": 1,
}
saveOpt = {
    "dpi": 100,
    "bbox_inches": "tight",
    "pad_inches": 0,
}


def plotInputPointCloud(result):
    eval.config = result["config"]
    points, colors = eval.getPointCloud(
        result["frame"],
        result["dataSetPath"],
        segmentationMethod="unfiltered",
    )
    fig, ax = setupLatexPlot3D()
    # set axis properties
    plt.axis("off")
    ax.set_xlim(-1 / styleOpt["zoomFactor"], 1 / styleOpt["zoomFactor"])
    ax.set_ylim(-1 / styleOpt["zoomFactor"], 1 / styleOpt["zoomFactor"])
    ax.set_zlim(-1 / styleOpt["zoomFactor"], 1 / styleOpt["zoomFactor"])

    # scale point set to fit in figure optimally
    x_max = np.max(points[:, 0])
    x_min = np.min(points[:, 0])
    y_max = np.max(points[:, 1])
    y_min = np.min(points[:, 1])
    z_max = np.max(points[:, 2])
    z_min = np.min(points[:, 2])
    centroid = np.array(
        [
            0.5 * (x_max + x_min),
            0.5 * (y_max + y_min),
            0.5 * (z_max + z_min),
        ]
    )
    points_centered = points - centroid
    scaling_factor = np.max(np.abs(points_centered))
    max_distance = np.max(np.linalg.norm(points_centered, axis=1))
    points_scaled = points_centered / scaling_factor
    plotPointCloud(
        ax=ax,
        points=points_scaled,
        colors=colors,
        size=styleOpt["pointCloudSize"],
    )
    ax.set_position(
        [
            styleOpt["subplot_position_left"],
            styleOpt["subplot_position_bottom"],
            styleOpt["subplot_position_right"],
            styleOpt["subplot_position_top"],
        ]
    )
    ax.view_init(elev=styleOpt["elevation"], azim=styleOpt["azimuth"])
    return fig, ax


def plotSegmentedPointCloud(result):
    eval.config = result["config"]
    points, colors = eval.getPointCloud(
        result["frame"],
        result["dataSetPath"],
    )
    fig, ax = setupLatexPlot3D()
    # set axis properties
    plt.axis("off")
    ax.set_xlim(-1 / styleOpt["zoomFactor"], 1 / styleOpt["zoomFactor"])
    ax.set_ylim(-1 / styleOpt["zoomFactor"], 1 / styleOpt["zoomFactor"])
    ax.set_zlim(-1 / styleOpt["zoomFactor"], 1 / styleOpt["zoomFactor"])

    # scale point set to fit in figure optimally
    x_max = np.max(points[:, 0])
    x_min = np.min(points[:, 0])
    y_max = np.max(points[:, 1])
    y_min = np.min(points[:, 1])
    z_max = np.max(points[:, 2])
    z_min = np.min(points[:, 2])
    centroid = np.array(
        [
            0.5 * (x_max + x_min),
            0.5 * (y_max + y_min),
            0.5 * (z_max + z_min),
        ]
    )
    points_centered = points - centroid
    scaling_factor = np.max(np.abs(points_centered))
    max_distance = np.max(np.linalg.norm(points_centered, axis=1))
    points_scaled = points_centered / scaling_factor
    plotPointCloud(
        ax=ax,
        points=points_scaled,
        colors=colors,
        size=styleOpt["pointCloudSize"],
    )
    ax.set_position(
        [
            styleOpt["subplot_position_left"],
            styleOpt["subplot_position_bottom"],
            styleOpt["subplot_position_right"],
            styleOpt["subplot_position_top"],
        ]
    )
    ax.view_init(elev=styleOpt["elevation"], azim=styleOpt["azimuth"])
    return fig, ax


def makePlot2D(result):
    fig, ax = setupLatexPlot3D()
    if controlOpt["visualizeInputPointCloud"]:
        eval.config = result["config"]
        Y, _ = eval.getPointCloud(result["frame"], result["dataSetPath"])
        plotPointSet(
            ax=ax,
            X=Y,
            size=styleOpt["pointCloudSize"],
            color=styleOpt["pointCloudColor"],
            alpha=styleOpt["pointCloudAlpha"],
            markerStyle=styleOpt["pointCloudMarkerStyle"],
            edgeColor=styleOpt["pointCloudEdgeColor"],
        )
    ax = eval.plotLocalizationResult3D(ax=ax, result=result)
    if styleOpt["elevation"] and styleOpt["azimuth"] is not None:
        ax.view_init(elev=styleOpt["elevation"], azim=styleOpt["azimuth"])
    return fig, ax


if __name__ == "__main__":
    if controlOpt["dataSetsToLoad"][0] == -1:
        dataSetsToEvaluate = resultFolderPaths
    else:
        dataSetsToEvaluate = [
            dataSetPath
            for i, dataSetPath in enumerate(resultFolderPaths)
            if i in controlOpt["dataSetsToLoad"]
        ]
    # load results
    for resultFolderPath in dataSetsToEvaluate:
        if controlOpt["resultsToLoad"][0] == -1:
            resultFiles = eval.list_result_files(resultFolderPath)
        else:
            resultFiles = eval.list_result_files(resultFolderPath)
            resultFiles = [
                file
                for i, file in enumerate(resultFiles)
                if i in controlOpt["resultsToLoad"]
            ]
        failedFrames = []
        for resultFile in resultFiles:
            resultFilePath = os.path.join(resultFolderPath, resultFile)
            result = eval.loadResults(resultFilePath)
            try:
                # get 2D Image
                img = eval.getImage(result["frame"], result["dataSetPath"])
                eval.plotImageWithMatplotlib(img)

                if controlOpt["plotInputPointCloud"]:
                    # get segmented point cloud
                    fig_in, ax_in = plotInputPointCloud(result)

                if controlOpt["plotSegmentedPointCloud"]:
                    # get segmented point cloud
                    fig_seg, ax_seg = plotSegmentedPointCloud(result)

                # save images
                if controlOpt["save"]:
                    id = "_".join(resultFile.split("_")[0:3])
                    fileNameImg = id + "_" + controlOpt["saveNameImg"]
                    fileNameInputPC = id + "_" + controlOpt["saveNameInputPointCloud"]
                    fileNameSegmentedPC = (
                        id + "_" + controlOpt["saveNameSegmentedPointCloud"]
                    )
                    dataSetName = result["dataSetPath"].split("/")[-2]
                    folderPath = os.path.join(controlOpt["saveFolder"], dataSetName)
                    savePathImg = os.path.join(folderPath, fileNameImg)
                    savePathInputPC = os.path.join(folderPath, fileNameInputPC)
                    savePathSegmentedPC = os.path.join(folderPath, fileNameSegmentedPC)
                    if not os.path.exists(folderPath):
                        os.makedirs(folderPath, exist_ok=True)
                    eval.saveImage(img, savePathImg)
                    if controlOpt["plotInputPointCloud"]:
                        fig_in.savefig(
                            savePathInputPC,
                            dpi=saveOpt["dpi"],
                            bbox_inches=saveOpt["bbox_inches"],
                            pad_inches=saveOpt["pad_inches"],
                        )
                    if controlOpt["plotSegmentedPointCloud"]:
                        fig_seg.savefig(
                            savePathSegmentedPC,
                            dpi=saveOpt["dpi"],
                            bbox_inches=saveOpt["bbox_inches"],
                            pad_inches=saveOpt["pad_inches"],
                        )
                    if controlOpt["saveAsPGF"]:
                        raise NotImplementedError
                        # plt.savefig(filePath, format="pgf", bbox_inches="tight", pad_inches=0)
                    if controlOpt["verbose"]:
                        print(
                            "Saved input image of result {} at {}.".format(
                                resultFile, savePathImg
                            )
                        )
                        print(
                            "Saved input point cloud of result {} at {}.".format(
                                resultFile, savePathInputPC
                            )
                        )
                        print(
                            "Saved segmented point cloud of result {} at {}.".format(
                                resultFile, savePathSegmentedPC
                            )
                        )
                if controlOpt["showPlot"]:
                    plt.show(block=controlOpt["block"])
                plt.close("all")
            except:
                failedFrames.append(result["frame"])
                traceback.print_exc()
        if len(failedFrames) > 0:
            print("Failed on frames {}".format(failedFrames))
