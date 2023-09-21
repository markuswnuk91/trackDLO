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
    "dataSetsToLoad": [0, 1, 2, 3, 4, 5],  # [0],[1],[2], [3],[4], [5]
    "resultsToLoad": [[0], [0], [27], [0], [10], [0]],  # [0],[0],[27],[0],[10], [0]
    "save": True,
    "saveAsPGF": False,
    "showInputImage": False,
    "showPlots": True,
    "plotTopologyExtractionResult": False,
    "plotCorrespondanceEstimationResult": False,
    "plotLocalizationResult3D": False,
    "plotLocalizationResult2D": True,
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
    "topologyExtractionResult": {
        "zoomFactor": 1.5,
        "plotEnvironment": True,
        "highlightWireHarness": False,
        "colorPalette": thesisColorPalettes["blues"],
        "colorPaletteStartValue": 0.5,
        "colorPaletteEndValue": 1,
        "leafNodeSize": 10,
        "leafNodeMarker": "s",
        "leafNodeAlpha": 0.1,
        "branchNodeSize": 10,
        "branchNodeMarker": "^",
        "branchNodeAlpha": 0.1,
        "azimuth": 91,
        "elevation": 50,
    },
    "correspondanceEstimationResult": {
        "xRotAngle": 0,
        "yRotAngle": -np.pi / 2,
        "zRotAngle": np.pi / 2,
        "xOffset": -0.6,
        "colorPalette": thesisColorPalettes["viridis"],
        "extractedTopologyColor": thesisColors["blue"],
        "correspondanceAlpha": 0.3,
        "templatePointSize": 5,
        "extractedPointSize": 5,
        "zoomFactor": 1.5,
        "azimuth": 90,
        "elevation": 90,
    },
    "localizationResult3D": {
        "modelPointSize": 3,
        "modelLineWidth": 1.7,
        "pointCloudSize": 3,
        "pointCloudColor": thesisColors["red"],
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
    eval.config = result["config"]
    pointCloud, colors = eval.getPointCloud(
        result["frame"],
        result["dataSetPath"],
        segmentationMethod="unfiltered",
    )
    segmentedPointCloud, _ = eval.getPointCloud(
        result["frame"],
        result["dataSetPath"],
    )
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
    zoom = styleOpt["topologyExtractionResult"]["zoomFactor"]
    ax.set_xlim(-1 / zoom, 1 / zoom)
    ax.set_ylim(-1 / zoom, 1 / zoom)
    ax.set_zlim(-1 / zoom, 1 / zoom)

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
    pointCloud = (pointCloud - centroid) / scaling_factor
    segmentedPointCloud = (segmentedPointCloud - centroid) / scaling_factor
    ax.set_position(
        [
            styleOpt["subplot_position_left"],
            styleOpt["subplot_position_bottom"],
            styleOpt["subplot_position_right"],
            styleOpt["subplot_position_top"],
        ]
    )
    ax.view_init(
        elev=styleOpt["topologyExtractionResult"]["elevation"],
        azim=styleOpt["topologyExtractionResult"]["azimuth"],
    )

    # plot point Cloud
    if styleOpt["topologyExtractionResult"]["plotEnvironment"]:
        plotPointCloud(ax=ax, points=pointCloud, colors=colors, size=1, alpha=0.1)
    if styleOpt["topologyExtractionResult"]["highlightWireHarness"]:
        plotPointSet(
            ax=ax, X=segmentedPointCloud, color=thesisColors["red"], size=1, alpha=0.1
        )
    colorPalette = styleOpt["topologyExtractionResult"]["colorPalette"]
    colorPaletteStartValue = styleOpt["topologyExtractionResult"][
        "colorPaletteStartValue"
    ]
    colorPaletteEndValue = styleOpt["topologyExtractionResult"]["colorPaletteEndValue"]
    colorScaleCoordinates = np.linspace(
        colorPaletteStartValue, colorPaletteEndValue, numBranches
    )
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
            zOrder=1000,
        )
    leafNodeSize = styleOpt["topologyExtractionResult"]["leafNodeSize"]
    leafNodeMarker = styleOpt["topologyExtractionResult"]["leafNodeMarker"]
    leafNodeAlpha = styleOpt["topologyExtractionResult"]["leafNodeAlpha"]
    for leafNodeIdx in leafNodes:
        node = topology.getNodes()[leafNodeIdx]
        branch = topology.getBranchesFromNode(node)
        branchIndex = topology.getBranchIndices(branch)[0]
        plotSinglePoint(
            ax=ax,
            x=points[leafNodeIdx, :],
            color=branchColors[branchIndex],
            size=leafNodeSize,
            marker=leafNodeMarker,
            alpha=leafNodeAlpha,
            zOrder=1000,
        )

    branchNodeColor = colorPalette.to_rgba(colorPaletteStartValue)[:3]
    branchNodeSize = 10
    branchNodeAlpha = 0.1
    branchNodeMarker = "^"
    for branchNode in branchNodes:
        plotSinglePoint(
            ax=ax,
            x=points[branchNode, :],
            color=branchNodeColor,
            size=branchNodeSize,
            marker=branchNodeMarker,
            alpha=branchNodeAlpha,
            zOrder=1000,
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
    xRotAngle = styleOpt["correspondanceEstimationResult"]["xRotAngle"]
    yRotAngle = styleOpt["correspondanceEstimationResult"]["yRotAngle"]
    zRotAngle = styleOpt["correspondanceEstimationResult"]["zRotAngle"]
    q = templateTopology.getGeneralizedCoordinates()
    q[0:3] = templateTopology.convertExtrinsicEulerAnglesToBallJointPositions(
        xRotAngle=xRotAngle, yRotAngle=yRotAngle, zRotAngle=zRotAngle
    )
    # shift model into position
    meanTargets = np.mean(targetPoints, axis=0)
    meanTemplate = np.mean(templateTopology.computeForwardKinematics(q)[0], axis=0)
    offset = styleOpt["correspondanceEstimationResult"]["xOffset"]
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

    colorPalette = styleOpt["correspondanceEstimationResult"]["colorPalette"]
    colorScaleCoordinates = np.linspace(0, 1, numBranches)
    branchColors = []
    for s in colorScaleCoordinates:
        branchColors.append(colorPalette.to_rgba(s)[:3])

    extractedTopologyColor = styleOpt["correspondanceEstimationResult"][
        "extractedTopologyColor"
    ]
    correspondanceAlpha = styleOpt["correspondanceEstimationResult"][
        "correspondanceAlpha"
    ]
    templatePointSize = styleOpt["correspondanceEstimationResult"]["templatePointSize"]
    extractedPointSize = styleOpt["correspondanceEstimationResult"][
        "extractedPointSize"
    ]
    plotTopology3D(ax=ax, topology=extractedTopology, color=extractedTopologyColor)
    plotBranchWiseColoredTopology3D(
        ax=ax,
        topology=templateTopology,
        colorPalette=colorPalette,
        pointSize=0.1,
    )
    for branchIndex in range(0, numBranches):
        correspondingTemplatePointIndices = np.where(np.array(B) == branchIndex)[0]
        correspondingTemplatePoints = templateTopologySamplePoints[
            correspondingTemplatePointIndices, :
        ]
        correspondingTargetPoints = targetPoints[correspondingTemplatePointIndices, :]
        plotCorrespondances3D(
            ax=ax,
            X=correspondingTemplatePoints,
            Y=correspondingTargetPoints,
            C=np.eye(len(correspondingTemplatePoints)),
            xColor=branchColors[branchIndex],
            yColor=branchColors[branchIndex],
            correspondanceColor=branchColors[branchIndex],
            lineAlpha=correspondanceAlpha,
            xSize=templatePointSize,
            ySize=extractedPointSize,
        )

    # set axis properties
    plt.axis("off")
    points = np.vstack((templateTopologySamplePoints, targetPoints))
    x_max = np.max(points[:, 0])
    x_min = np.min(points[:, 0])
    y_max = np.max(points[:, 1])
    y_min = np.min(points[:, 1])
    z_max = np.max(points[:, 2])
    z_min = np.min(points[:, 2])
    centroid = np.mean(
        np.array(
            [
                [x_min, y_min, z_min],
                [x_min, y_min, z_max],
                [x_min, y_max, z_max],
                [x_min, y_max, z_min],
                [x_max, y_min, z_min],
                [x_max, y_min, z_max],
                [x_max, y_max, z_max],
                [x_max, y_max, z_min],
            ]
        ),
        axis=0,
    )
    max_dist = np.max(np.array([x_max - x_min, y_max - y_min, z_max - z_min]))
    zoom = styleOpt["correspondanceEstimationResult"]["zoomFactor"]
    ax.set_xlim(
        (centroid[0] - max_dist / 2) / zoom, (centroid[0] + max_dist / 2) / zoom
    )
    ax.set_ylim(
        (centroid[1] - max_dist / 2) / zoom, (centroid[1] + max_dist / 2) / zoom
    )
    ax.set_zlim(
        (centroid[2] - max_dist / 2) / zoom, (centroid[2] + max_dist / 2) / zoom
    )
    ax.set_position(
        [
            styleOpt["subplot_position_left"],
            styleOpt["subplot_position_bottom"],
            styleOpt["subplot_position_right"],
            styleOpt["subplot_position_top"],
        ]
    )
    ax.view_init(
        elev=styleOpt["correspondanceEstimationResult"]["elevation"],
        azim=styleOpt["correspondanceEstimationResult"]["azimuth"],
    )
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
    colorPalette = thesisColorPalettes["viridis"]
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
    for i, resultFolderPath in enumerate(dataSetsToEvaluate):
        if controlOpt["resultsToLoad"][i] == -1:
            resultFiles = eval.list_result_files(resultFolderPath)
        else:
            resultFiles = eval.list_result_files(resultFolderPath)
            resultFiles = [
                file
                for n, file in enumerate(resultFiles)
                if n in controlOpt["resultsToLoad"][i]
            ]

        failedFrames = []
        for i, resultFile in enumerate(resultFiles):
            resultFilePath = os.path.join(resultFolderPath, resultFile)
            result = eval.loadResults(resultFilePath)
            id = "_".join(resultFile.split("_")[0:3])
            dataSetName = result["dataSetPath"].split("/")[-2]
            folderPath = os.path.join(saveOpt["saveFolder"], dataSetName)
            if controlOpt["save"] and not os.path.exists(folderPath):
                os.makedirs(folderPath, exist_ok=True)
            try:
                # get 2D Image
                inputImg = eval.getImage(result["frame"], result["dataSetPath"])
                if controlOpt["showInputImage"]:
                    eval.plotImageWithMatplotlib(inputImg)
                if controlOpt["save"]:
                    fileName = id + "_" + saveOpt["saveName_inputImg"]
                    savePath = os.path.join(folderPath, fileName)
                    eval.saveImage(inputImg, savePath)
                if controlOpt["plotTopologyExtractionResult"]:
                    # get segmented point cloud
                    (
                        fig_topologyExtraction,
                        ax_topologyExtraction,
                    ) = plotTopologyExtractionResult(result)
                    if controlOpt["showPlots"]:
                        fig_topologyExtraction.show()
                    if controlOpt["save"]:
                        fileName = id + "_" + saveOpt["saveName_topologyExtraction"]
                        savePath = os.path.join(folderPath, fileName)
                        if controlOpt["saveAsPGF"]:
                            raise NotImplementedError
                        else:
                            fig_topologyExtraction.savefig(
                                savePath,
                                dpi=saveOpt["dpi"],
                                bbox_inches=saveOpt["bbox_inches"],
                                pad_inches=saveOpt["pad_inches"],
                            )
                        if controlOpt["verbose"]:
                            print(
                                "Saved topology extraction result {}/{} at {}.".format(
                                    i + 1, len(resultFiles), savePath
                                )
                            )
                if controlOpt["plotCorrespondanceEstimationResult"]:
                    # get segmented point cloud
                    (
                        fig_correspondanceEstimation,
                        ax_correspondanceEstimation,
                    ) = plotCorrespondanceEstimationResult(result)
                    if controlOpt["showPlots"]:
                        fig_correspondanceEstimation.show()
                    if controlOpt["save"]:
                        fileName = (
                            id + "_" + saveOpt["saveName_correspondanceEstimation"]
                        )
                        savePath = os.path.join(folderPath, fileName)
                        if controlOpt["saveAsPGF"]:
                            raise NotImplementedError
                        else:
                            fig_correspondanceEstimation.savefig(
                                savePath,
                                dpi=saveOpt["dpi"],
                                bbox_inches=saveOpt["bbox_inches"],
                                pad_inches=saveOpt["pad_inches"],
                            )
                        if controlOpt["verbose"]:
                            print(
                                "Saved topology extraction result {}/{} at {}.".format(
                                    i + 1, len(resultFiles), savePath
                                )
                            )
                if controlOpt["plotLocalizationResult3D"]:
                    fig_localization3D, ax_localization3D = plotLocalizationResult3D(
                        result
                    )
                    if controlOpt["showPlots"]:
                        fig_localization3D.show()
                    if controlOpt["save"]:
                        fileName = id + "_" + saveOpt["saveName_localizationResult3D"]
                        savePath = os.path.join(folderPath, fileName)
                        if controlOpt["saveAsPGF"]:
                            raise NotImplementedError
                        else:
                            fig_localization3D.savefig(
                                savePath,
                                dpi=saveOpt["dpi"],
                                bbox_inches=saveOpt["bbox_inches"],
                                pad_inches=saveOpt["pad_inches"],
                            )
                        if controlOpt["verbose"]:
                            print(
                                "Saved topology extraction result {}/{} at {}.".format(
                                    i + 1, len(resultFiles), savePath
                                )
                            )
                if controlOpt["plotLocalizationResult2D"]:
                    localizationResult2DImg = (
                        eval.plotBranchWiseColoredLocalizationResult2D(result)
                    )
                    localizationResult2DImg
                    if controlOpt["showPlots"]:
                        eval.plotImageWithMatplotlib(
                            localizationResult2DImg, block=True
                        )
                    if controlOpt["save"]:
                        fileName = id + "_" + saveOpt["saveName_localizationResult2D"]
                        savePath = os.path.join(folderPath, fileName)
                        eval.saveImage(localizationResult2DImg, savePath)
                plt.close("all")
            except:
                failedFrames.append(result["frame"])
                traceback.print_exc()
        if len(failedFrames) > 0:
            print("Failed on frames {}".format(failedFrames))
