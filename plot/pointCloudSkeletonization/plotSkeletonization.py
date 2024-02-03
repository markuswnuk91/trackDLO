import sys
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

try:
    sys.path.append(os.getcwd().replace("/plot", ""))
    from src.evaluation.evaluation import Evaluation
    from src.visualization.plot3D import *
    from src.visualization.plotUtils import scale_axes_to_fit
except:
    print("Imports for plotting images for 3D skeletonization failed.")
    raise
save = True
visualizeIterations = False
relFilePath = "data/darus_data_download/data/20230516_Configurations_labeled/20230516_115857_arena/data/20230516_120332_090647_image_rgb.png"
saveFolderPath = "imgs/skeletonization"
elev = 47
azim = 108
paramters = {
    "numSeedPoints": 30,
    "h": 0.01,
    "hAnnealing": 0.8,
    "hMin": 0.002,
    "mu": 0.35,
    "max_iterations": 100,
    "densityCompensation": 0,
}
downsampling_nth_element = 10
if __name__ == "__main__":
    fileName = os.path.basename(relFilePath)
    dataFolderPath = os.path.dirname(relFilePath)
    dataSetFolderPath = os.path.dirname(dataFolderPath) + "/"
    eval = Evaluation("plot/pointCloudSkeletonization/evalConfig_skeletonizaiton.json")
    frame = eval.getFrameFromFileName(dataSetFolderPath, fileName)
    pointCloud = eval.getPointCloud(
        frame, dataSetFolderPath, segmentationMethod="standard"
    )
    points = pointCloud[0][::downsampling_nth_element, :]
    colors = pointCloud[1][::downsampling_nth_element, :]
    result_l1 = eval.runL1Median(
        pointSet=points,
        visualizeIterations=visualizeIterations,
        l1Parameters=paramters,
    )
    # plot input
    fig, ax = setupLatexPlot3D()
    plotPointSet(ax=ax, X=points, color=[0, 0, 0], size=1, alpha=0.5)
    # plotPointSet(ax=ax, X=points, color=colors, size=5, alpha=0.3)
    scale_axes_to_fit(ax=ax, points=points, zoom=2)
    ax.view_init(elev=elev, azim=azim)
    plt.axis("off")
    if save:
        plt.savefig(
            os.path.join(saveFolderPath, "pointCloud"),
            bbox_inches="tight",
            pad_inches=0,
            dpi=300,
        )
    # plot result
    fig, ax = setupLatexPlot3D()
    plotPointSet(ax=ax, X=points, color=[0.1, 0.1, 0.1], size=1, alpha=0.01)
    plotPointSet(ax=ax, X=result_l1["T"], color=[1, 0, 0], size=10, alpha=1, zOrder=3)
    scale_axes_to_fit(ax=ax, points=points, zoom=2)
    ax.view_init(elev=elev, azim=azim)
    plt.axis("off")
    if save:
        plt.savefig(
            os.path.join(saveFolderPath, "skeleton"),
            bbox_inches="tight",
            pad_inches=0,
            dpi=300,
        )
    # plot magnified area
    zoom = 3.5
    offset = np.array([1.3, 0.5, 0])
    fig, ax = setupLatexPlot3D()
    xlim = (0.35931863545790393, 0.5080058107363304)
    ylim = (0.0335721155968565, 0.1822592908752829)
    zlim = (0.0832218455732201, 0.23190902085164747)
    plotPointSet(ax=ax, X=points, color=[0, 0, 0], size=1, alpha=0.05)
    # plotPointSet(ax=ax, X=result_l1["T"], color=[1, 0, 0], size=10, alpha=1, zOrder=3)
    for p in result_l1["T"]:
        plotPoint(ax=ax, x=p, color=[1, 0, 0], size=25, zOrder=3)
    scale_axes_to_fit(ax=ax, points=points)
    ax.view_init(elev=elev, azim=azim)
    # ax.set_xlim((np.array(ax.get_xlim()) + offset[0]) / zoom)
    # ax.set_ylim((np.array(ax.get_ylim()) + offset[1]) / zoom)
    # ax.set_zlim((np.array(ax.get_zlim()) + offset[2]) / zoom)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    plt.axis("off")
    if save:
        plt.savefig(
            os.path.join(saveFolderPath, "magnification"),
            bbox_inches="tight",
            pad_inches=0,
            dpi=300,
        )

    plt.show(block=True)
    print(ax.get_xlim())
    print(ax.get_ylim())
    print(ax.get_zlim())
    print("Done")
