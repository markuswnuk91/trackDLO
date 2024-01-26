import sys
import os
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

try:
    sys.path.append(os.getcwd().replace("/experimental", ""))
    from src.sensing.dataHandler import DataHandler
except:
    print("Imports failed.")
    raise

relFilePath_Img = "data/darus_data_download/data/20230516_Configurations_labeled/20230516_115857_arena/data/20230516_121354_510965_image_rgb.png"
n_bins = 100  # Number of bins in the colormap
max_cutoff_val = 180
min_cutoff_val = 70
invalid_val = 150
if __name__ == "__main__":
    # load image
    dataHandler = DataHandler()
    fileName = os.path.basename(relFilePath_Img)
    dirPath = os.path.dirname(os.path.dirname(relFilePath_Img)) + "/"
    disp = dataHandler.loadStereoDataSet(fileName, dirPath)[1]

    # find invalid values and set them -1
    invalid_vals = set()
    disp[np.where(disp == 511.875)] = 0

    # disp_img_normalized = 1 / np.max(disp) * disp
    # disp_img_normalized = disp_img_normalized.astype(np.uint8)
    disp_img_normalized = (disp - min_cutoff_val) / (max_cutoff_val - min_cutoff_val)
    cmap = plt.colormaps["rainbow"]
    # cmap = LinearSegmentedColormap.from_list("red_to_blue", colors, N=n_bins)
    colored_disp_img = cmap(disp_img_normalized)
    colored_disp_img[disp == 0] = invalid_val

    # Convert RGBA to BGR for OpenCV
    colored_disp_img_rgb = (colored_disp_img[:, :, :3] * 255).astype(np.uint8)
    colored_disp_img_bgr = cv2.cvtColor(colored_disp_img_rgb, cv2.COLOR_RGB2BGR)
    # cv2.imshow("Disparity Image", colored_disp_img_bgr)
    plt.imshow(colored_disp_img_rgb, vmin=0, vmax=1, cmap=plt.colormaps["rainbow_r"])
    plt.axis("off")
    cbar = plt.colorbar(ticks=[0, 1])
    cbar.set_label("Disparity", labelpad=-20)
    cbar.ax.set_yticklabels(["High", "Low"])
    plt.show()
    cv2.waitKey(0)
    print("DOne")
