import sys
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

try:
    sys.path.append(os.getcwd().replace("/plot", ""))
    from src.sensing.dataHandler import DataHandler
    from src.sensing.preProcessing import PreProcessing
    from src.evaluation.evaluation import Evaluation
    from src.visualization.plot3D import *
    from src.visualization.plotUtils import scale_axes_to_fit
    from src.localization.downsampling.filter.lofFilter import LocalOutlierFactorFilter
except:
    print("Imports for plotting point cloud processing methods failed.")
    raise

# control parameters
save = True
show = True
zoom = 2.0
pointSize = 10
outlierSize = 30
elev = 42
azim = 100
# relFilePath = "data/darus_data_download/data/20230516_Configurations_labeled/20230516_115857_arena/data/20230516_121354_510965_image_rgb.png"
relFilePath = "data/darus_data_download/data/20230516_Configurations_labeled/20230516_115857_arena/data/20230516_120332_090647_image_rgb.png"
configFilePath = "plot/pointCloudProcessing/evalConfig_lofFiltering.json"
saveFolderPath = "imgs/localOutlierFactor"
if __name__ == "__main__":
    fileName = os.path.basename(relFilePath)
    dataFolderPath = os.path.dirname(relFilePath)
    dataSetFolderPath = os.path.dirname(dataFolderPath) + "/"
    eval = Evaluation(configFilePath=configFilePath)
    frame = eval.getFrameFromFileName(dataSetFolderPath, fileName)
    pointCloud = eval.getPointCloud(
        frame, dataSetFolderPath, segmentationMethod="standard"
    )

    # input
    fig, ax = setupLatexPlot3D()
    points = pointCloud[0]
    points = points - np.mean(points, axis=0)
    colors = pointCloud[1]
    scale_axes_to_fit(ax, points, zoom=zoom)
    ax.view_init(elev=elev, azim=azim)
    plt.axis("off")
    plotPointCloud(
        ax=ax,
        points=points,
        colors=colors,
        size=pointSize,
        alpha=0.5,
        markerStyle=".",
    )
    if save:
        plt.savefig(
            os.path.join(saveFolderPath, "input"),
            bbox_inches="tight",
            pad_inches=0,
            dpi=300,
        )
    # filtered
    fig, ax = setupLatexPlot3D()
    scale_axes_to_fit(ax, points, zoom=zoom)
    ax.view_init(elev=elev, azim=azim)
    plt.axis("off")
    lof = LocalOutlierFactorFilter(numNeighbors=150, contamination=0.1)
    filteredPointSet = lof.sampleLOF(points)
    inlierIndices = lof.inlierIndices
    outliers = lof.Outliers
    # plotPointCloud(
    #     ax=ax,
    #     points=filteredPointSet,
    #     colors=colors[inlierIndices, :],
    #     size=1,
    #     alpha=0.5,
    #     markerStyle=".",
    # )
    plotPointSet(
        ax=ax,
        X=points,
        color=[0, 0, 0],
        size=pointSize,
        alpha=0.5,
        markerStyle=".",
    )
    plotPointSet(ax=ax, X=outliers, color=[1, 0, 0], alpha=1, size=outlierSize)
    if save:
        plt.savefig(
            os.path.join(saveFolderPath, "outliers"),
            bbox_inches="tight",
            pad_inches=0,
            dpi=300,
        )

    # result
    fig, ax = setupLatexPlot3D()
    plotPointSet(
        ax=ax,
        X=points[inlierIndices, :],
        color=colors[inlierIndices, :],
        size=pointSize,
        alpha=0.5,
        markerStyle=".",
    )
    scale_axes_to_fit(ax, points, zoom=zoom)
    ax.view_init(elev=elev, azim=azim)
    plt.axis("off")
    if save:
        plt.savefig(
            os.path.join(saveFolderPath, "result"),
            bbox_inches="tight",
            pad_inches=0,
            dpi=300,
        )
    print("Done")
    if show:
        plt.show(block=True)
