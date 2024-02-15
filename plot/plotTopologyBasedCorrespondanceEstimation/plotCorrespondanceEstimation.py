import sys
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

try:
    sys.path.append(os.getcwd().replace("/plot", ""))
    from src.evaluation.initialLocalization.initialLocalizationEvaluation import (
        InitialLocalizationEvaluation,
    )
    from src.visualization.plot3D import *
    from src.visualization.colors import *
    from src.visualization.plotUtils import scale_axes_to_fit
except:
    print("Imports for plotting tolology extraction failed.")
    raise
runOpt = {"save": True, "runTopologyExtraction": False}
visOpt = {"visLocalizationIterations": True}
saveOpt = {
    "correspondanceEstimationResultPath": "data/plots/topologyBasedCorrespondanceEstimation",
    "saveFolderPath": "imgs/topologyBasedCorrespondanceEstimation",
    "dpi": 300,
}
relFilePath = "data/darus_data_download/data/20230516_Configurations_labeled/20230516_115857_arena/data/20230516_120332_090647_image_rgb.png"
saveFolderPath = "imgs/topologyReconstruction"
configPath = "plot/plotTopologyReconstruction/plotConfig.json"
topologyExtractionResultPath = ""
colorMaps = {
    "red": plt.cm.ScalarMappable(
        cmap=matplotlib.colormaps["Reds"],
        norm=matplotlib.colors.Normalize(vmin=-1, vmax=2),
    ),
    "blue": plt.cm.ScalarMappable(
        cmap=matplotlib.colormaps["Blues"],
        norm=matplotlib.colors.Normalize(vmin=-1, vmax=2),
    ),
    "viridis": plt.cm.ScalarMappable(
        cmap=matplotlib.colormaps["viridis"],
        norm=matplotlib.colors.Normalize(vmin=0, vmax=1),
    ),
}
styleOpt = {
    "subplot_position_left": 0,
    "subplot_position_bottom": 0,
    "subplot_position_right": 1,
    "subplot_position_top": 1,
    "zoomFactor": 1.5,
    "azimuth": -179,
    "elevation": -90,
    "pointCloudSize": 1,
    "topologyModels": {
        "extractedTopologyColorPalette": colorMaps["red"],
        "templateTopologyColorPalette": colorMaps["blue"],
        "templatePointSize": 0.1,
        "extractedPointSize": 0.1,
    },
    "sampledTopologies": {
        "extractedTopologyColorPalette": colorMaps["red"],
        "templateTopologyColorPalette": colorMaps["blue"],
        "sampleColorTemplate": [0, 0, 1],
        "sampleColorExtracted": [1, 0, 0],
        "templatePointSize": 0.1,
        "extractedPointSize": 0.1,
        "templateSampleSize": 20,
        "extractedSampleSize": 20,
    },
    "correspondanceEstimationResult": {
        "xRotAngle": 0,
        "yRotAngle": -np.pi / 2,
        "zRotAngle": np.pi / 2,
        "xOffset": -0.6,
        "extractedTopologyColorPalette": colorMaps["red"],
        "templateTopologyColorPalette": colorMaps["blue"],
        "correspondanceColorPalette": colorMaps["viridis"],
        "correspondanceAlpha": 0.3,
        "templatePointSize": 0.1,
        "extractedPointSize": 0.1,
        "correspondancePointSize": 20,
        "correspondanceAlpha": 0.5,
    },
}


def plotExtractedTopology(topology, samplePoints):
    fig, ax = setupLatexPlot3D()
    plotBranchWiseColoredTopology3D(
        ax=ax,
        topology=topology,
        colorPalette=styleOpt["topologyModels"]["extractedTopologyColorPalette"],
        pointSize=styleOpt["topologyModels"]["extractedPointSize"],
    )
    plotPointSet(
        ax=ax,
        X=samplePoints,
        color=styleOpt["sampledTopologies"]["sampleColorExtracted"],
        size=styleOpt["sampledTopologies"]["extractedSampleSize"],
    )
    return fig, ax


def plotTemplateTopology(topology, samplePoints):
    fig, ax = setupLatexPlot3D()
    plotBranchWiseColoredTopology3D(
        ax=ax,
        topology=topology,
        colorPalette=styleOpt["topologyModels"]["templateTopologyColorPalette"],
        pointSize=styleOpt["topologyModels"]["templatePointSize"],
    )
    plotPointSet(
        ax=ax,
        X=samplePoints,
        color=styleOpt["sampledTopologies"]["sampleColorTemplate"],
        size=styleOpt["sampledTopologies"]["templateSampleSize"],
    )
    return fig, ax


def plotTopologyModels(templateTopology, extractedTopology):
    fig, ax = setupLatexPlot3D()
    plotBranchWiseColoredTopology3D(
        ax=ax,
        topology=templateTopology,
        colorPalette=styleOpt["topologyModels"]["templateTopologyColorPalette"],
        pointSize=styleOpt["topologyModels"]["templatePointSize"],
        lineWidth=3,
    )
    plotBranchWiseColoredTopology3D(
        ax=ax,
        topology=extractedTopology,
        colorPalette=styleOpt["topologyModels"]["extractedTopologyColorPalette"],
        pointSize=styleOpt["topologyModels"]["extractedPointSize"],
    )
    return fig, ax


def plotCorrespondanceEstimationSamples(
    templateTopology, extractedTopology, samplePointsTemplate, samplePointsExtracted
):
    fig, ax = setupLatexPlot3D()
    plotBranchWiseColoredTopology3D(
        ax=ax,
        topology=templateTopology,
        colorPalette=styleOpt["sampledTopologies"]["templateTopologyColorPalette"],
        pointSize=styleOpt["sampledTopologies"]["extractedPointSize"],
        lineWidth=3,
    )
    plotPointSet(
        ax=ax,
        X=samplePointsTemplate,
        color=styleOpt["sampledTopologies"]["sampleColorTemplate"],
        size=styleOpt["sampledTopologies"]["templateSampleSize"],
    )
    plotBranchWiseColoredTopology3D(
        ax=ax,
        topology=extractedTopology,
        colorPalette=styleOpt["sampledTopologies"]["extractedTopologyColorPalette"],
        pointSize=styleOpt["sampledTopologies"]["extractedPointSize"],
    )
    plotPointSet(
        ax=ax,
        X=samplePointsExtracted,
        color=styleOpt["sampledTopologies"]["sampleColorExtracted"],
        size=styleOpt["sampledTopologies"]["extractedSampleSize"],
    )
    return fig, ax


def plotCorrespondanceEstimationResult(templateTopology, extractedTopology):
    fig, ax = setupLatexPlot3D()
    numBranches = templateTopology.getNumBranches()

    correspondanceAlpha = styleOpt["correspondanceEstimationResult"][
        "correspondanceAlpha"
    ]
    templatePointSize = styleOpt["correspondanceEstimationResult"]["templatePointSize"]
    extractedPointSize = styleOpt["correspondanceEstimationResult"][
        "extractedPointSize"
    ]
    plotBranchWiseColoredTopology3D(
        ax=ax,
        topology=templateTopology,
        colorPalette=styleOpt["correspondanceEstimationResult"][
            "templateTopologyColorPalette"
        ],
        pointSize=styleOpt["correspondanceEstimationResult"]["templatePointSize"],
        lineWidth=3,
    )
    plotBranchWiseColoredTopology3D(
        ax=ax,
        topology=extractedTopology,
        colorPalette=styleOpt["correspondanceEstimationResult"][
            "extractedTopologyColorPalette"
        ],
        pointSize=styleOpt["correspondanceEstimationResult"]["extractedPointSize"],
    )

    for branchIndex in range(0, numBranches):
        correspondingTemplatePointIndices = np.where(np.array(B) == branchIndex)[0]
        correspondingTemplatePoints = templateTopologySamplePoints[
            correspondingTemplatePointIndices, :
        ]
        correspondingTargetPoints = targetPoints[correspondingTemplatePointIndices, :]

        for i, (x, y) in enumerate(
            zip(correspondingTemplatePoints, correspondingTargetPoints)
        ):
            s = i / len(correspondingTemplatePoints)
            correspondanceColor = styleOpt["correspondanceEstimationResult"][
                "correspondanceColorPalette"
            ].to_rgba(s)[:3]
            plotLine(
                ax=ax,
                pointPair=np.vstack((x, y)),
                color=correspondanceColor,
                alpha=styleOpt["correspondanceEstimationResult"]["correspondanceAlpha"],
            )
            plotPoint(
                ax=ax,
                x=x,
                color=correspondanceColor,
                size=styleOpt["correspondanceEstimationResult"][
                    "correspondancePointSize"
                ],
            )
            plotPoint(
                ax=ax,
                x=y,
                color=correspondanceColor,
                size=styleOpt["correspondanceEstimationResult"][
                    "correspondancePointSize"
                ],
            )
        # plotCorrespondances3D(
        #     ax=ax,
        #     X=correspondingTemplatePoints,
        #     Y=correspondingTargetPoints,
        #     C=np.eye(len(correspondingTemplatePoints)),
        #     xColor=correspondanceColor,
        #     yColor=correspondanceColor,
        #     correspondanceColor=correspondanceColor
        #     lineAlpha=correspondanceAlpha,
        #     xSize=templatePointSize,
        #     ySize=extractedPointSize,
        # )
    return fig, ax


def scale_axes(ax, points):
    # set axis properties
    plt.axis("off")
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
    zoom = styleOpt["zoomFactor"]
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
        elev=styleOpt["elevation"],
        azim=styleOpt["azimuth"],
    )
    return ax


if __name__ == "__main__":
    eval = InitialLocalizationEvaluation(
        configFilePath="plot/plotTopologyBasedCorrespondanceEstimation/evalConfig.json"
    )

    if runOpt["runTopologyExtraction"]:
        # perfrom topology extraction & correspondance estimation for input data set
        fileName = os.path.basename(relFilePath)
        dataFolderPath = os.path.dirname(relFilePath)
        dataSetPath = os.path.dirname(dataFolderPath) + "/"
        frame = eval.getFrameFromFileName(dataSetPath, fileName)
        pointCloud = eval.getPointCloud(
            frame, dataSetPath, segmentationMethod="standard"
        )
        Y = pointCloud[0]
        model, modelParameters = eval.getModel(dataSetPath)
        extractedTopology = eval.extractMinimumSpanningTreeTopology(Y, model)[
            "extractedTopology"
        ]
        localizationResult, _ = eval.runInitialLocalization(
            pointSet=Y,
            extractedTopology=extractedTopology,
            bdloModelParameters=modelParameters,
            visualizeCorresponanceEstimation=True,
            visualizeIterations=visOpt["visLocalizationIterations"],
            visualizeResult=False,
            block=False,
            closeAfterRunning=True,
            logResults=True,
        )
        eval.saveWithPickle(
            data=localizationResult,
            filePath=os.path.join(
                saveOpt["correspondanceEstimationResultPath"],
                "correspondanceEstimationResult.pkl",
            ),
            recursionLimit=10000,
        )
    else:
        # save correspondance estimation results
        localizationResult = eval.loadResults(
            os.path.join(
                saveOpt["correspondanceEstimationResultPath"],
                "correspondanceEstimationResult.pkl",
            )
        )

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

    # plot extracted topology
    fig, ax = plotExtractedTopology(
        topology=extractedTopology, samplePoints=targetPoints
    )
    scale_axes_to_fit(ax=ax, points=extractedTopology.X, zoom=styleOpt["zoomFactor"])
    ax.view_init(
        elev=styleOpt["elevation"],
        azim=styleOpt["azimuth"],
    )
    plt.axis("off")
    if runOpt["save"]:
        plt.savefig(
            os.path.join(saveOpt["saveFolderPath"], "ExtractedTopology"),
            bbox_inches="tight",
            pad_inches=0,
            dpi=saveOpt["dpi"],
        )
    # plot template topology
    fig, ax = plotTemplateTopology(
        topology=templateTopology, samplePoints=templateTopologySamplePoints
    )
    scale_axes(ax=ax, points=templateTopologySamplePoints)
    if runOpt["save"]:
        plt.savefig(
            os.path.join(saveOpt["saveFolderPath"], "TemplateTopology"),
            bbox_inches="tight",
            pad_inches=0,
            dpi=saveOpt["dpi"],
        )
    # # model topology vs reconstructed topology
    # fig, ax = plotTopologyModels(
    #     templateTopology=templateTopology, extractedTopology=extractedTopology
    # )
    # scale_axes(ax=ax, points=np.vstack((templateTopologySamplePoints, targetPoints)))

    # # sampled point sets on topologyies
    # fig, ax = plotCorrespondanceEstimationSamples(
    #     templateTopology,
    #     extractedTopology,
    #     samplePointsTemplate=templateTopologySamplePoints,
    #     samplePointsExtracted=extractedTopologySamplePoints,
    # )
    # scale_axes(ax=ax, points=np.vstack((templateTopologySamplePoints, targetPoints)))

    # create correspondance estimation plots
    fig, ax = plotCorrespondanceEstimationResult(
        templateTopology=templateTopology, extractedTopology=extractedTopology
    )
    scale_axes(ax=ax, points=np.vstack((templateTopologySamplePoints, targetPoints)))
    if runOpt["save"]:
        plt.savefig(
            os.path.join(saveOpt["saveFolderPath"], "Correspondances"),
            bbox_inches="tight",
            pad_inches=0,
            dpi=saveOpt["dpi"],
        )
    plt.show(block=True)
# correspondances between point sets
