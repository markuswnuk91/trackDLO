# plotting scipt for figure to visualize stereo projection. Plots:
# - Input Image
# - Colored disparity map
# - Point Cloud representation

import sys
import os
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

try:
    sys.path.append(os.getcwd().replace("/plot", ""))
    from src.sensing.dataHandler import DataHandler
    from src.sensing.preProcessing import PreProcessing
    from src.visualization.plot3D import *
except:
    print("Imports failed.")
    raise
save = True
show = False
relFilePath_Img = "data/darus_data_download/data/20230516_Configurations_labeled/20230516_115857_arena/data/20230516_121354_510965_image_rgb.png"
saveFolderPath = "imgs/stereoProjection"
max_cutoff_val = 180
min_cutoff_val = 70
invalid_val = 150
dpi = 300


textwidth_in_pt = 483.6969
figureScaling = 0.45
latexFontSize_in_pt = 20
latexFootNoteFontSize_in_pt = 10
desiredFigureWidth = figureScaling * textwidth_in_pt
desiredFigureHeight = figureScaling * textwidth_in_pt
tex_fonts = {
    #    "pgf.texsystem": "pdflatex",
    # Use LaTeX to write all text
    "text.usetex": True,
    "font.family": "serif",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": latexFontSize_in_pt,
    "font.size": latexFontSize_in_pt,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": latexFontSize_in_pt,
    "xtick.labelsize": latexFontSize_in_pt,
    "ytick.labelsize": latexFontSize_in_pt,
}
# matplotlib.use("pgf")
matplotlib.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "text.usetex": True,
        "pgf.rcfonts": False,
    }
)
matplotlib.rcParams.update(tex_fonts)

if __name__ == "__main__":
    # load image
    dataHandler = DataHandler()
    fileName = os.path.basename(relFilePath_Img)
    dirPath = os.path.dirname(os.path.dirname(relFilePath_Img)) + "/"
    img, disp = dataHandler.loadStereoDataSet(fileName, dirPath)

    if save:
        cv2.imwrite(
            os.path.join(saveFolderPath, "inputImage.png"),
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
        )

    # find invalid values and set them -1
    invalid_vals = set()
    disp[np.where(disp == 511.875)] = 0

    # disp_img_normalized = 1 / np.max(disp) * disp
    # disp_img_normalized = disp_img_normalized.astype(np.uint8)
    disp_img_normalized = (disp - min_cutoff_val) / (max_cutoff_val - min_cutoff_val)
    cmap = plt.colormaps["viridis"]
    # cmap = LinearSegmentedColormap.from_list("red_to_blue", colors, N=n_bins)
    colored_disp_img = cmap(disp_img_normalized)
    colored_disp_img[disp == 0] = invalid_val

    # Convert RGBA to BGR for OpenCV
    colored_disp_img_rgb = (colored_disp_img[:, :, :3] * 255).astype(np.uint8)
    colored_disp_img_bgr = cv2.cvtColor(colored_disp_img_rgb, cv2.COLOR_RGB2BGR)
    # cv2.imshow("Disparity Image", colored_disp_img_bgr)
    (img_height, image_width, _) = img.shape
    # plt.figure(figsize=(img_height / 100, image_width / 100), dpi=100)
    plt.imshow(colored_disp_img_rgb, vmin=0, vmax=1, cmap=plt.colormaps["viridis"])
    plt.axis("off")
    cbar = plt.colorbar(ticks=[0.05, 0.95])
    cbar.ax.tick_params(size=0)
    cbar.set_label("Disparity", labelpad=-20)
    cbar.ax.set_yticklabels(["Low", "High"])
    if save:
        plt.savefig(
            os.path.join(saveFolderPath, "disparityMap"),
            bbox_inches="tight",
            pad_inches=0,
            dpi=290,
        )
    if show:
        plt.show()
        cv2.waitKey(0)

    # generate point cloud
    preProcessor = PreProcessing(defaultLoadFolderPath=dirPath)

    # filter out all irrelevant points
    mask = preProcessor.getMaskFromRGB_applyHSVFilter(img,hMin=0,hMax=10, sMin=0, sMax=255, vMin=0, vMax= 255)

    pointCloud = preProcessor.calculatePointCloud(
        img, disp, preProcessor.cameraParameters["qmatrix"], mask=mask
    )
    points = pointCloud[0][::10, :]
    colors = pointCloud[1][::10, :]

    # transfrom points in robot coodinate sytem
    points = preProcessor.transformPointsFromCameraToRobotBaseCoordinates(points)
    bb_values = preProcessor.getBoundingBoxDefaultValues()
    bb_values["zMin"] = 0.2
    points, colors = preProcessor.getInliersFromBoundingBox((points, colors), bb_values)

    fig, ax = setupLatexPlot3D()
    plotPointCloud(ax=ax, points=points, colors=colors, size=3, markerStyle=".")
    zoom = 1.5
    # ax.set_xlim(0.08, 0.79)
    # ax.set_ylim(-0.25, 0.39)
    # ax.set_zlim(0.1, 0.79)
    ax.set_xlim(0.1669268301796496, 0.6379790523402699)
    ax.set_ylim(-0.12152774363714591, 0.3030827101414415)
    ax.set_zlim(0.0988528858789807, 0.5566360313590203)
    # # scale point set to fit in figure optimally
    # x_max = np.max(points[:, 0])
    # x_min = np.min(points[:, 0])
    # y_max = np.max(points[:, 1])
    # y_min = np.min(points[:, 1])
    # z_max = np.max(points[:, 2])
    # z_min = np.min(points[:, 2])
    # centroid = np.array(
    #     [
    #         0.5 * (x_max + x_min),
    #         0.5 * (y_max + y_min),
    #         0.5 * (z_max + z_min),
    #     ]
    # )
    # points_centered = points - centroid
    # scaling_factor = np.max(np.abs(points_centered))
    # max_distance = np.max(np.linalg.norm(points_centered, axis=1))
    # points = points_centered / scaling_factor
    # pointCloud = (pointCloud - centroid) / scaling_factor
    ax.set_position(
        [
            0,
            0,
            1,
            1,
        ]
    )
    ax.view_init(elev=24, azim=111)
    plt.axis("off")
    if save:
        plt.savefig(
            os.path.join(saveFolderPath, "pointCloud"),
            bbox_inches="tight",
            pad_inches=0,
            dpi=300,
        )
    plt.show()
    print("Done")
