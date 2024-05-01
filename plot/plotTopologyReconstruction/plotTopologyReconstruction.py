import sys
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

try:
    sys.path.append(os.getcwd().replace("/plot", ""))
    from src.evaluation.evaluation import Evaluation
    from src.visualization.plot3D import *
    from src.visualization.colors import *
    from src.visualization.plotUtils import scale_axes_to_fit
    from src.localization.topologyExtraction.minimalSpanningTreeExtraction import (
        MinimalSpanningTreeExtraction,
    )
    from src.utils.utils import minimalSpanningTree
except:
    print("Imports for plotting tolology extraction failed.")
    raise
save = True
relFilePath = "data/darus_data_download/data/20230516_Configurations_labeled/20230516_115857_arena/data/20230516_120332_090647_image_rgb.png"
saveFolderPath = "imgs/topologyReconstruction"
dpi = 300
configPath = "plot/plotTopologyReconstruction/plotConfig.json"
elev = -90
azim = 180
zoom = 1.5
downsampling_nth_element = 50
nPaths = 4
colorPalette = plt.cm.ScalarMappable(
    cmap=matplotlib.colormaps["Greys"],
    norm=matplotlib.colors.Normalize(vmin=0, vmax=1),
)
min_color_coordinate = 0.1
max_color_coordinate = 1
outlier_color = [0.5, 0.5, 0.5]
outlier_color = [1, 0, 0]
# lineStyles = ["-", "--", "-.", ":"]
lineStyle = "-"
text_scaling = 1.8

if __name__ == "__main__":
    fileName = os.path.basename(relFilePath)
    dataFolderPath = os.path.dirname(relFilePath)
    dataSetFolderPath = os.path.dirname(dataFolderPath) + "/"
    eval = Evaluation(configPath)
    frame = eval.getFrameFromFileName(dataSetFolderPath, fileName)
    pointCloud = eval.getPointCloud(
        frame, dataSetFolderPath, segmentationMethod="standard"
    )
    points = pointCloud[0][::downsampling_nth_element, :]
    colors = pointCloud[1][::downsampling_nth_element, :]

    topologyReconstruction = MinimalSpanningTreeExtraction(X=points, nPaths=nPaths)
    A_mst = topologyReconstruction.getMinimalSpanningTreeAdjacencyMatrix(X=points)
    shortestPaths = topologyReconstruction.findNLongestPaths(A_mst, nPaths)
    extractedTopology = topologyReconstruction.extractTopology(X=points, nPaths=nPaths)

    # plot input point set
    ax = plt.figure(figsize=(10, 5)).add_subplot(projection="3d")
    plotPointSet(ax=ax, X=points, color=[0, 0, 0], size=5)
    scale_axes_to_fit(ax=ax, points=points, zoom=zoom)
    ax.view_init(azim=azim, elev=elev)
    plt.axis("off")
    if save:
        plt.savefig(
            os.path.join(saveFolderPath, "Input"),
            bbox_inches="tight",
            pad_inches=0,
            dpi=dpi,
        )

    # plot MST
    ax = plt.figure(figsize=(10, 5)).add_subplot(projection="3d")
    plotGraph3D(
        ax=ax,
        X=points,
        adjacencyMatrix=A_mst,
        pointColor=[0, 0, 0],
        pointAlpha=1,
        pointSize=0.1,
        lineColor=[0, 0, 0],
        lineAlpha=0.5,
    )
    scale_axes_to_fit(ax=ax, points=points, zoom=zoom)
    ax.view_init(azim=azim, elev=elev)
    plt.axis("off")
    if save:
        plt.savefig(
            os.path.join(saveFolderPath, "MST"),
            bbox_inches="tight",
            pad_inches=0,
            dpi=dpi,
        )

    # plot shorest paths
    set_text_to_latex_font(scale_text=text_scaling)
    ax = plt.figure(figsize=(10, 5)).add_subplot(projection="3d")
    plotGraph3D(
        ax=ax,
        X=points,
        adjacencyMatrix=A_mst,
        pointColor=[0, 0, 0],
        pointAlpha=0.1,
        pointSize=0.1,
        lineColor=outlier_color,
        lineWidth=1,
    )
    legendSymbols = []
    for i, path in enumerate(shortestPaths):
        color_coordinate = min_color_coordinate + (len(shortestPaths) - i) / len(
            shortestPaths
        ) * (max_color_coordinate - min_color_coordinate)
        color = colorPalette.to_rgba(color_coordinate)[:3]
        # plotPointSet(ax=ax, X=points, color=[0.5, 0.5, 0.5], size=0, alpha=0.1)
        # plotPointSet(ax=ax, X=points[path, :], color=[0.3, 0.3, 0.3], size=1)
        plotPointSet(ax=ax, X=points[path, :], color=color, size=0.1)
        for i, pointIndex in enumerate(path[:-1]):
            plotLine(
                ax=ax,
                pointPair=np.vstack((points[pointIndex, :], points[path[i + 1], :])),
                color=color,
                lineStyle=lineStyle,
            )
    markerConfig_1 = Line2D(
            [],
            [],
            color=color,
            linestyle="-",
        )
    markerConfig_2 = 
    markerCorrespondance = markerConfig_1 = Line2D(
            [],
            [],
            color=color,
            linestyle="-",
        )
    legendSymbols = [markerConfig_1, markerConfig_2, markerCorrespondance]
        legendSymbols.append(legendSymbol)
    legendSymbols.append(
        Line2D(
            [],
            [],
            color=outlier_color,
            linestyle="-",
        )
    )
    ax.legend(
        handles=legendSymbols,
        labels=[
            "1st",
            "2nd",
            "3rd",
            "4th",
            "outliers",
        ],
        # loc="upper right",
        loc="upper center",
        ncol=2,
        columnspacing=1,
        handletextpad=1,
    )
    scale_axes_to_fit(ax=ax, points=points, zoom=zoom)
    ax.view_init(azim=azim, elev=elev)
    plt.axis("off")
    if save:
        plt.savefig(
            os.path.join(saveFolderPath, "Result"),
            bbox_inches="tight",
            pad_inches=0,
            dpi=dpi,
        )

    # # plot legend
    # plt.figure(figsize=(3, 6))
    # ax.legend(
    #     handles=legendSymbols,
    #     labels=[
    #         "1st longest path",
    #         "2nd longest path",
    #         "3rd longest path",
    #         "4th longest path",
    #         "outliers",
    #     ],
    #     # loc="upper right",
    #     nrow=2,
    # )

# plot extracted connectivity graph
ax = plt.figure(figsize=(10, 5)).add_subplot(projection="3d")
plotGraph3D(
    ax=ax,
    X=extractedTopology.X,
    adjacencyMatrix=extractedTopology.adjacencyMatrix,
    pointColor=[0, 0, 0],
    pointAlpha=1,
    pointSize=5,
    lineColor=[0, 0, 0],
    lineAlpha=0.5,
)
scale_axes_to_fit(ax=ax, points=points, zoom=zoom)
ax.view_init(azim=azim, elev=elev)
plt.axis("off")
if save:
    plt.savefig(
        os.path.join(saveFolderPath, "ConnectivityGraph"),
        bbox_inches="tight",
        pad_inches=0,
        dpi=dpi,
    )
# plot extracted leaf and branch nodes
ax = plt.figure(figsize=(10, 5)).add_subplot(projection="3d")
plotGraph3D(
    ax=ax,
    X=extractedTopology.X,
    adjacencyMatrix=extractedTopology.adjacencyMatrix,
    pointColor=[0, 0, 0],
    pointAlpha=1,
    pointSize=5,
    lineColor=[0, 0, 0],
    lineAlpha=0.5,
)
# branch Nodes
branchNodeColor = [1, 0, 0]
branchNodeIndices = extractedTopology.getBranchNodeIndices()
plotPointSet(
    ax=ax,
    X=extractedTopology.X[branchNodeIndices, :],
    color=branchNodeColor,
    size=500,
    alpha=0.1,
)
plotPointSet(
    ax=ax,
    X=extractedTopology.X[branchNodeIndices, :],
    color=branchNodeColor,
    size=30,
    alpha=1,
)
# leaf nodes
leafNodeColor = [0, 0, 1]
leafNodeIndices = extractedTopology.getLeafNodeIndices()
plotPointSet(
    ax=ax,
    X=extractedTopology.X[leafNodeIndices, :],
    color=leafNodeColor,
    size=500,
    alpha=0.1,
)
plotPointSet(
    ax=ax,
    X=extractedTopology.X[leafNodeIndices, :],
    color=leafNodeColor,
    size=30,
    alpha=1,
)
legendSymbols.append(
    Line2D(
        [],
        [],
        color=outlier_color,
        linestyle="-",
    )
)
leafNodeSymbol = Line2D(
    [],
    [],
    color=leafNodeColor,
    marker="o",
    linestyle="None",
    markersize=10,
)
branchNodeSymbol = Line2D(
    [],
    [],
    color=branchNodeColor,
    marker="o",
    linestyle="None",
    markersize=10,
)
ax.legend(
    handles=[leafNodeSymbol, branchNodeSymbol],
    labels=[
        "leaf node",
        "branch node",
    ],
    # loc="upper right",
    loc="upper center",
    ncol=2,
    columnspacing=0.1,
    handletextpad=0.05,
)
scale_axes_to_fit(ax=ax, points=points, zoom=zoom)
ax.view_init(azim=azim, elev=elev)
plt.axis("off")
if save:
    plt.savefig(
        os.path.join(saveFolderPath, "LeafBranchNodeExtraction"),
        bbox_inches="tight",
        pad_inches=0,
        dpi=dpi,
    )

# plot extracted topology
ax = plt.figure(figsize=(10, 5)).add_subplot(projection="3d")
plotBranchWiseColoredTopology3D(ax=ax, topology=extractedTopology, pointSize=3)
scale_axes_to_fit(ax=ax, points=points, zoom=zoom)
ax.view_init(azim=azim, elev=elev)
plt.axis("off")
if save:
    plt.savefig(
        os.path.join(saveFolderPath, "ExtractedTopology"),
        bbox_inches="tight",
        pad_inches=0,
        dpi=dpi,
    )

plt.show(block=True)
print("Done")
