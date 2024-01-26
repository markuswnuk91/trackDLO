from skimage.filters import frangi, hessian
import sys
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

try:
    sys.path.append(os.getcwd().replace("/experimental", ""))
    from src.sensing.dataHandler import DataHandler
    from src.sensing.imageProcessing import ImageProcessing
except:
    print("Imports for plotting image processing methods failed.")
    raise

# control parameters
relFilePath_Img = "data/darus_data_download/data/20230524_152039_ManipulationSequences_mountedWireHarness_arena/data/20230524_152143_314549_image_rgb.png"

if __name__ == "__main__":
    dataHandler = DataHandler()

    # load image
    img = dataHandler.loadImage(relFilePath_Img)

    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # apply filter
    filter_img = frangi(img_gray, alpha=50, beta=10, gamma=20)
    cv2.imshow(
        "Color filter input",
        cv2.resize(
            filter_img,
            None,
            fx=0.25,
            fy=0.25,
        ),
    )
    cv2.waitKey(0)
