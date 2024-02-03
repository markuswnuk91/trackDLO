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
show = {"input": False, "downsampled": False, "boxfiltered": False}
zoom = 2.0

relFilePath = "data/darus_data_download/data/20230516_Configurations_labeled/20230516_115857_arena/data/20230516_121354_510965_image_rgb.png"
configFilePath = "plot/plotdata/plotLocalOutlierFilter/evalConfig.json"
saveFolderPath = "imgs/pointCloudProcessing"


def get_axis_limits(points):
    x_max = np.max(points[:, 0])
    x_min = np.min(points[:, 0])
    y_max = np.max(points[:, 1])
    y_min = np.min(points[:, 1])
    z_max = np.max(points[:, 2])
    z_min = np.min(points[:, 2])
    return {
        "xMin": x_min,
        "xMax": x_max,
        "yMin": y_min,
        "yMax": y_max,
        "zMin": z_min,
        "zMax": z_max,
    }


def center_point_cloud(points):
    centroid = np.mean(points, axis=0)
    return points - centroid


def scale_axes_to_fit(ax, points, zoom=1):
    ax.set_position(
        [
            0,
            0,
            1,
            1,
        ]
    )
    axis_limits = get_axis_limits(points)
    ax_range = np.max(
        np.array(
            (
                axis_limits["xMax"] - axis_limits["xMin"],
                axis_limits["yMax"] - axis_limits["yMin"],
                axis_limits["zMax"] - axis_limits["zMin"],
            )
        )
    )
    centroid = np.mean(points, axis=0)
    ax_offset = 2 * zoom
    ax.set_xlim(centroid[0] - ax_range / ax_offset, centroid[0] + ax_range / ax_offset)
    ax.set_ylim(centroid[1] - ax_range / ax_offset, centroid[1] + ax_range / ax_offset)
    ax.set_zlim(centroid[2] - ax_range / ax_offset, centroid[2] + ax_range / ax_offset)
    return ax


if __name__ == "__main__":
    fileName = os.path.basename(relFilePath)
    dataFolderPath = os.path.dirname(relFilePath)
    dataSetFolderPath = os.path.dirname(dataFolderPath) + "/"

    dataHandler = DataHandler()
    preProcessor = PreProcessing(defaultLoadFolderPath=dataSetFolderPath)
    img, disp = dataHandler.loadStereoDataSet(fileName, dataSetFolderPath)
    pointCloud = preProcessor.calculatePointCloud(
        img, disp, preProcessor.cameraParameters["qmatrix"]
    )
    points = pointCloud[0][::3, :]
    colors = pointCloud[1][::3, :]
    points = preProcessor.transformPointsFromCameraToRobotBaseCoordinates(points)

    # prefilter point cloud
    bb_values = preProcessor.getBoundingBoxDefaultValues()
    # bb_values["xMin"] = -2
    # bb_values["xMax"] = 2
    # bb_values["yMin"] = -2
    # bb_values["yMax"] = 2
    # bb_values["zMin"] = -2
    # bb_values["zMax"] = 2
    points, colors = preProcessor.getInliersFromBoundingBox((points, colors), bb_values)

    # gernerate input point cloud
    # centroid = np.array(
    #     [
    #         0.5 * (x_max - x_min),
    #         0.5 * (y_max - y_min),
    #         0.5 * (z_max - z_min),
    #     ]
    # )
    centroid = np.mean(points, axis=0)
    points = points - centroid
    # scaling_factor = np.max(np.abs(points_centered))
    # max_distance = np.max(np.linalg.norm(points_centered, axis=1))
    # points = points_centered / scaling_factor
    # pointCloud = (pointCloud - centroid) / scaling_factor
    fig, ax = setupLatexPlot3D()
    scale_axes_to_fit(ax, points, zoom=zoom)
    ax.view_init(elev=24, azim=111)
    plt.axis("off")
    plotPointCloud(
        ax=ax, points=points, colors=colors, size=3, alpha=0.1, markerStyle="."
    )
    if show["input"]:
        plt.show(block=False)
    if save:
        plt.savefig(
            os.path.join(saveFolderPath, "pointCloud"),
            bbox_inches="tight",
            pad_inches=0,
            dpi=300,
        )
    # generate uniform downsample plot
    nth_element = 5
    points_downsampled = points[::nth_element, :]
    colors_downsampled = colors[::nth_element, :]
    fig, ax = setupLatexPlot3D()
    scale_axes_to_fit(ax, points_downsampled, zoom=zoom)
    ax.view_init(elev=24, azim=111)
    plt.axis("off")
    plotPointCloud(
        ax=ax,
        points=points_downsampled,
        colors=colors_downsampled,
        size=3,
        alpha=0.1,
        markerStyle=".",
    )
    if show["downsampled"]:
        plt.show(block=False)
    if save:
        plt.savefig(
            os.path.join(saveFolderPath, "pointCloud_downsampled"),
            bbox_inches="tight",
            pad_inches=0,
            dpi=300,
        )
    # gernerate box filter plot
    bounding_boxlimits = {
        "xMin": -0.2,
        "xMax": 0.25,
        "yMin": -0.2,
        "yMax": 0.2,
        "zMin": -0.02,
        "zMax": 0.35,
    }
    inliers, inlier_colors = preProcessor.getInliersFromBoundingBox(
        (points, colors), bounding_boxlimits
    )
    outliers, outlier_colors = preProcessor.getOutliersFromBoundingBox(
        (points, colors), bounding_boxlimits
    )
    outlier_colors[:, :] = np.array([0.5, 0.5, 0.5])
    fig, ax = setupLatexPlot3D()
    ax = scale_axes_to_fit(ax, points, zoom=zoom)
    plotPointSet(ax=ax, X=outliers, color=outlier_colors, alpha=0.1, size=1)
    plotPointSet(ax=ax, X=inliers, color=inlier_colors, alpha=0.1, size=1)

    plotCube(
        ax,
        bounding_boxlimits["xMin"],
        bounding_boxlimits["xMax"],
        bounding_boxlimits["yMin"],
        bounding_boxlimits["yMax"],
        bounding_boxlimits["zMin"],
        bounding_boxlimits["zMax"],
        color="r",
        alpha=0.03,
    )
    ax.view_init(elev=24, azim=111)
    plt.axis("off")
    if save:
        plt.savefig(
            os.path.join(saveFolderPath, "pointCloud_boxfiltered"),
            bbox_inches="tight",
            pad_inches=0,
            dpi=300,
        )
    if show["boxfiltered"]:
        plt.show(block=True)
