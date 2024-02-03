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
    from src.localization.downsampling.fps.farthestPointSampling import FPS
except:
    print("Imports for plotting images for 3D skeletonization failed.")
    raise
save = True
visualizeIterations = False
run = {"l1": True, "som": True, "fps": True}
relFilePaths = [
    # "data/darus_data_download/data/20230516_Configurations_labeled/20230516_112207_YShape/data/20230516_112236_148986_image_rgb.png",
    # "data/darus_data_download/data/20230516_Configurations_labeled/20230516_113957_Partial/data/20230516_114110_904273_image_rgb.png",
    "data/darus_data_download/data/20230516_Configurations_labeled/20230516_115857_arena/data/20230516_120332_090647_image_rgb.png",
]
numSamples = [300]
index = 0
relFilePath = relFilePaths[index]
saveFolderPath = "imgs/skeletonizationMethodComparison"
elev = 47
azim = 108
zoom = 2
if __name__ == "__main__":
    som_results = []
    l1_results = []
    pointClouds = []
    som_figures = []
    l1_figures = []
    input_figures = []
    for i, _ in enumerate(numSamples):
        fileName = os.path.basename(relFilePath)
        dataFolderPath = os.path.dirname(relFilePath)
        dataSetFolderPath = os.path.dirname(dataFolderPath) + "/"
        eval = Evaluation(
            "plot/pointCloudSkeletonization/evalConfig_skeletonizaiton.json"
        )
        frame = eval.getFrameFromFileName(dataSetFolderPath, fileName)
        pointCloud = eval.getPointCloud(
            frame, dataSetFolderPath, segmentationMethod="standard"
        )
        points = pointCloud[0]
        colors = pointCloud[1]
        pointClouds.append(pointCloud)

        if run["l1"]:
            l1_parameters = eval.config["topologyExtraction"]["l1Parameters"]
            l1_parameters["numSeedPoints"] = numSamples[i]
            result_l1 = eval.runL1Median(
                pointSet=points,
                visualizeIterations=visualizeIterations,
                l1Parameters=l1_parameters,
            )
            l1_results.append(result_l1)

        if run["som"]:
            som_parameters = eval.config["topologyExtraction"]["somParameters"]
            som_parameters["numSeedPoints"] = numSamples[i]
            result_som = eval.runSOM(
                pointSet=points,
                visualizeIterations=visualizeIterations,
                somParameters=som_parameters,
            )
            som_results.append(result_som)

        if run["fps"]:
            fps = FPS(points, numSamples[i])
            result_fps = fps.fit()

        # plot input
        fig, ax = setupLatexPlot3D()
        # plotPointSet(ax=ax, X=points, color=[0, 0, 0], size=1, alpha=0.5)
        plotPointSet(ax=ax, X=points, color=colors, size=1, alpha=0.1)
        scale_axes_to_fit(ax=ax, points=points)
        ax.view_init(elev=elev, azim=azim)
        input_figures.append(fig)

        # plot results
        if run["l1"]:
            fig, ax = setupLatexPlot3D()
            plotPointSet(ax=ax, X=points, color=[0.5, 0.5, 0.5], size=5, alpha=0.01)
            plotPointSet(
                ax=ax, X=result_l1["T"], color=[1, 0, 0], size=10, alpha=1, zOrder=3
            )
            scale_axes_to_fit(ax=ax, points=points, zoom=zoom)
            ax.view_init(elev=elev, azim=azim)
            plt.axis("off")
            l1_results.append(fig)
            if save:
                plt.savefig(
                    os.path.join(saveFolderPath, "skeleton_l1"),
                    bbox_inches="tight",
                    pad_inches=0,
                    dpi=300,
                )
        if run["som"]:
            fig, ax = setupLatexPlot3D()
            plotPointSet(ax=ax, X=points, color=[0.5, 0.5, 0.5], size=5, alpha=0.01)
            # plotPointSet(
            #     ax=ax, X=result_som["T"], color=[1, 0, 0], size=10, alpha=1, zOrder=3
            # )
            for p in result_som["T"]:
                plotPoint(ax=ax, x=p, color=[1, 0, 0], size=10, zOrder=3)
            scale_axes_to_fit(ax=ax, points=points, zoom=zoom)
            ax.view_init(elev=elev, azim=azim)
            plt.axis("off")
            som_results.append(fig)
            if save:
                plt.savefig(
                    os.path.join(saveFolderPath, "skeleton_som"),
                    bbox_inches="tight",
                    pad_inches=0,
                    dpi=300,
                )
        if run["fps"]:
            fig, ax = setupLatexPlot3D()
            plotPointSet(ax=ax, X=points, color=[0.5, 0.5, 0.5], size=5, alpha=0.01)
            # plotPointSet(ax=ax, X=result_fps, color=[1, 0, 0], size=10, alpha=1)
            for p in result_fps:
                plotPoint(ax=ax, x=p, color=[1, 0, 0], size=10, zOrder=3)
            scale_axes_to_fit(ax=ax, points=points, zoom=zoom)
            ax.view_init(elev=elev, azim=azim)
            plt.axis("off")
            som_results.append(fig)

            if save:
                plt.savefig(
                    os.path.join(saveFolderPath, "skeleton_fps"),
                    bbox_inches="tight",
                    pad_inches=0,
                    dpi=300,
                )
        plt.show(block=True)
    print("Done")
