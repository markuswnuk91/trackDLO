import sys
import os
import matplotlib.pyplot as plt
import numpy as np
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
    "dataSetsToLoad": [2],  # [4]
    "resultsToLoad": [3],  # [0]
    "save": False,
    "showInputImage": False,
    "block": False,
    "plotSkeletonizationResult": False,
    "plotTopologyExtractionResult": False,
    "plotCorrespondanceEstimationResult": False,
    "plotLocalizationResult3D": True,
    "verbose": True,
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
    "localizationResult3D": {
        "modelPointSize": 3,
        "modelLineWidth": 1.7,
        "pointCloudSize": 3,
        "pointCloudColor": colors["red"],
        "pointCloudAlpha": 0.5,
        "pointCloudMarkerStyle": "o",
        "pointCloudDownSampleFactor": 7,
        "makeBackgroundGray": True,
        "backgroundColor": [0.75, 0.75, 0.75],
        "azimuth": 70,
        "elevation": 30,
        "axisLimX": [0.0875, 0.725],
        "axisLimY": [-0.22, 0.43],
        "axisLimZ": [0.2, 0.75],
    },
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
    "saveFolder": "data/eval/initialLocalization/plots/intermediateResults",
    "saveName_inputImg": "inputImg",
    "saveName_segmentedPointCloud": "segmentedPointCloud",
    "saveName_skeletonization": "skeltonizationResult",
    "saveName_topologyExtraction": "topologyExtractionResult",
    "saveName_correspondanceEstimation": "correspondanceEstimationResult",
    "saveName_localizationResult2D": "localizationResult2D",
    "saveName_localizationResult3D": "localizationResult3D",
    "dpi": 100,
    "bbox_inches": "tight",
    "pad_inches": 0,
}


def plotTopologyExtractionResult(result):
    topologyExtractionResult = result["topologyExtractionResult"]
    topology = topologyExtractionResult["extractedTopology"]
    points = topology.X
    leafNodes = topology.getLeafNodeIndices()
    branchNodes = topology.getBranchNodeIndices()
    featureMatrix = topology.featureMatrix
    numBranches = topology.getNumBranches()
    branchwiseIndices = []
    for branch in topology.getBranches():
        correspondingIndices = []
        for node in branch.getNodes():
            nodeIndex = topology.getNodeIndex(node)
            correspondingIndices.append(nodeIndex)
        branchwiseIndices.append(correspondingIndices)

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
    points = points_centered / scaling_factor

    ax.set_position(
        [
            styleOpt["subplot_position_left"],
            styleOpt["subplot_position_bottom"],
            styleOpt["subplot_position_right"],
            styleOpt["subplot_position_top"],
        ]
    )
    ax.view_init(elev=styleOpt["elevation"], azim=styleOpt["azimuth"])

    colorPalette = colorPalettes["viridis"]
    colorScaleCoordinates = np.linspace(0, 1, numBranches)
    branchColors = []
    for s in colorScaleCoordinates:
        branchColors.append(colorPalette.to_rgba(s)[:3])
    for branchIdx, nodeIndices in enumerate(branchwiseIndices):
        adjacencyMatrix = np.diag(np.ones(len(nodeIndices) - 1), 1) + np.diag(
            np.ones(len(nodeIndices) - 1), -1
        )
        plotGraph3D(
            ax=ax,
            X=points[nodeIndices, :],
            adjacencyMatrix=adjacencyMatrix,
            pointColor=branchColors[branchIdx],
            lineColor=branchColors[branchIdx],
        )
    leafNodeColor = [0, 1, 0]
    leafNodeSize = 30
    leafNodeMarker = "s"
    for leafNodeIdx in leafNodes:
        node = topology.getNodes()[leafNodeIdx]
        branch = topology.getBranchesFromNode(node)
        branchIndex = topology.getBranchIndices(branch)[0]
        plotPoint(
            ax=ax,
            x=points[leafNodeIdx, :],
            color=branchColors[branchIndex],
            size=leafNodeSize,
            marker=leafNodeMarker,
        )

    branchNodeColor = colorPalette.to_rgba(0)[:3]
    branchNodeSize = 40
    branchNodeMarker = "^"
    for branchNode in branchNodes:
        plotPoint(
            ax=ax,
            x=points[branchNode, :],
            color=branchNodeColor,
            size=branchNodeSize,
            marker=branchNodeMarker,
        )

    return fig, ax


def plotCorrespondanceEstimationResult(result):
    localizationResult = result["localizationResult"]
    topology = localizationResult["extractedTopology"]
    modelParameters = localizationResult["modelParameters"]
    C = localizationResult["C"]
    S = localizationResult["S"]
    extractedTopologySamplePoints = localizationResult["YTarget"]
    templateTopology = eval.generateModel(modelParameters)
    extractedTopology = localizationResult["extractedTopology"]
    targetPoints = C.T @ extractedTopologySamplePoints
    templateTopologySamplePoints = []
    # orient the model
    q = templateTopology.getGeneralizedCoordinates()
    q[0:3] = templateTopology.convertExtrinsicEulerAnglesToBallJointPositions(
        xRotAngle=0, yRotAngle=np.pi / 2, zRotAngle=np.pi / 2
    )
    # shift model into position
    meanTargets = np.mean(targetPoints, axis=0)
    meanTemplate = np.mean(templateTopology.computeForwardKinematics(q)[0], axis=0)
    offset = -0.6
    q[3] = q[3] + (meanTargets - meanTemplate)[0] + offset
    q[4] = q[4] + (meanTargets - meanTemplate)[1]
    q[5] = q[5] + (meanTargets - meanTemplate)[2]
    templateTopology.setGeneralizedCoordinates(q)
    (
        templateTopologySamplePoints,
        B,
    ) = templateTopology.samplePointsForCorrespondanceEstimation(q, S)

    fig, ax = setupLatexPlot3D()
    numBranches = templateTopology.getNumBranches()

    colorPalette = colorPalettes["viridis"]
    colorScaleCoordinates = np.linspace(0, 1, numBranches)
    branchColors = []
    for s in colorScaleCoordinates:
        branchColors.append(colorPalette.to_rgba(s)[:3])

    extractedTopologyColor = colors["blue"]
    for branchIndex in range(0, numBranches):
        correspondingTemplatePointIndices = np.where(np.array(B) == branchIndex)[0]
        correspondingTemplatePoints = templateTopologySamplePoints[
            correspondingTemplatePointIndices, :
        ]
        correspondingTargetPoints = targetPoints[correspondingTemplatePointIndices, :]
        plotTopology3D(ax=ax, topology=extractedTopology, color=extractedTopologyColor)
        plotBranchWiseColoredTopology3D(
            ax=ax, topology=templateTopology, colorPalette=colorPalette
        )
        correspondanceAlpha = 0.3
        plotCorrespondances3D(
            ax=ax,
            X=correspondingTemplatePoints,
            Y=correspondingTargetPoints,
            C=np.eye(len(correspondingTemplatePoints)),
            xColor=branchColors[branchIndex],
            yColor=branchColors[branchIndex],
            correspondanceColor=branchColors[branchIndex],
            lineAlpha=correspondanceAlpha,
            ySize=5,
        )
    plt.show(block=True)
    return fig, ax


def plotLocalizationResult3D(result):
    fig, ax = setupLatexPlot3D()
    eval.config = result["config"]
    Y, colors = eval.getPointCloud(
        result["frame"],
        result["dataSetPath"],
        segmentationMethod="unfiltered",
    )
    # downsample point cloud
    Y = Y[:: styleOpt["localizationResult3D"]["pointCloudDownSampleFactor"], :]
    colors = colors[
        :: styleOpt["localizationResult3D"]["pointCloudDownSampleFactor"], :
    ]
    # remove values with low z value
    keepIdxs = np.where(Y[:, 2] >= 0.25)[0]
    Y = Y[keepIdxs, :]
    colors = colors[keepIdxs, :]

    # color background gray
    if styleOpt["localizationResult3D"]["makeBackgroundGray"]:
        colors[
            ((colors[:, 0] <= 0.7) | (colors[:, 1] >= 0.5) & (colors[:, 2] >= 0.5))
        ] = np.array(styleOpt["localizationResult3D"]["backgroundColor"])
    plotPointCloud(
        ax=ax,
        points=Y,
        colors=colors,
        size=styleOpt["localizationResult3D"]["pointCloudSize"],
        alpha=styleOpt["localizationResult3D"]["pointCloudAlpha"],
    )
    # plotPointSet(
    #     ax=ax,
    #     X=Y,
    #     size=styleOpt["localizationResult3D"]["pointCloudSize"],
    #     color=styleOpt["localizationResult3D"]["pointCloudColor"],
    #     alpha=styleOpt["localizationResult3D"]["pointCloudAlpha"],
    #     markerStyle=styleOpt["localizationResult3D"]["pointCloudMarkerStyle"],
    # )
    modelParameters = result["modelParameters"]
    model = eval.generateModel(modelParameters)
    q = result["localizationResult"]["q"]
    model.setGeneralizedCoordinates(q)
    colorPalette = colorPalettes["viridis"]
    plotBranchWiseColoredTopology3D(
        ax=ax,
        topology=model,
        colorPalette=colorPalette,
        lineWidth=styleOpt["localizationResult3D"]["modelLineWidth"],
        pointSize=styleOpt["localizationResult3D"]["modelPointSize"],
        zOrder=1000,
    )
    ax.set_xlim(
        styleOpt["localizationResult3D"]["axisLimX"][0],
        styleOpt["localizationResult3D"]["axisLimX"][1],
    )
    ax.set_ylim(
        styleOpt["localizationResult3D"]["axisLimY"][0],
        styleOpt["localizationResult3D"]["axisLimY"][1],
    )
    ax.set_zlim(
        styleOpt["localizationResult3D"]["axisLimZ"][0],
        styleOpt["localizationResult3D"]["axisLimZ"][1],
    )
    ax.view_init(
        elev=styleOpt["localizationResult3D"]["elevation"],
        azim=styleOpt["localizationResult3D"]["azimuth"],
    )
    ax.set_position(
        [
            styleOpt["subplot_position_left"],
            styleOpt["subplot_position_bottom"],
            styleOpt["subplot_position_right"],
            styleOpt["subplot_position_top"],
        ]
    )
    plt.show(block=True)
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
                if controlOpt["showInputImage"]:
                    eval.plotImageWithMatplotlib(img)

                if controlOpt["plotSkeletonizationResult"]:
                    raise NotImplementedError

                if controlOpt["plotTopologyExtractionResult"]:
                    # get segmented point cloud
                    (
                        fig_topologyExtraction,
                        ax_topologyExtraction,
                    ) = plotTopologyExtractionResult(result)

                if controlOpt["plotCorrespondanceEstimationResult"]:
                    # get segmented point cloud
                    (
                        fig_correspondanceEstimation,
                        ax_correspondanceEstimation,
                    ) = plotCorrespondanceEstimationResult(result)

                if controlOpt["plotLocalizationResult3D"]:
                    fig_localization3D, ax_localization3D = plotLocalizationResult3D(
                        result
                    )
                # save images
                if controlOpt["save"]:
                    id = "_".join(resultFile.split("_")[0:3])
                    fileNameImg = id + "_" + controlOpt["saveNameImg"]
                    fileNameInputPC = id + "_" + controlOpt["saveNameInputPointCloud"]
                    fileNameSegmentedPC = (
                        id + "_" + controlOpt["saveNameSegmentedPointCloud"]
                    )
                    dataSetName = result["dataSetPath"].split("/")[-2]
                    folderPath = os.path.join(saveOpt["saveFolder"], dataSetName)
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
