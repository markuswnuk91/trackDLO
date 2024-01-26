import sys
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

try:
    sys.path.append(os.getcwd().replace("/plot", ""))
    from src.sensing.dataHandler import DataHandler
    from src.sensing.imageProcessing import ImageProcessing
except:
    print("Imports for plotting image processing methods failed.")
    raise

# control parameters
save = True
show = True

relFilePath_Img = "data/darus_data_download/data/20230511_HandOcclusion/20230511_104424_Arena/data/20230511_104530_448898_image_rgb.png"
relFilePath_Background = "data/darus_data_download/data/20230511_132621_Background/data/20230511_132621_368078_image_rgb.png"
saveFolderPath = "imgs/plots_imageProcessingMethods/"


# roi filter values
uMin = 0.3
uMax = 0.93
vMin = 0.05
vMax = 0.9

if __name__ == "__main__":
    dataHandler = DataHandler()
    imageProcessor = ImageProcessing()

    # load image
    roi_filter_img_rgb = dataHandler.loadImage(relFilePath_Img)

    # apply roi filter
    roi_filter_mask = imageProcessor.getMaskFromRGB_applyROI(
        roi_filter_img_rgb, uMin, uMax, vMin, vMax
    )
    roi_filter_result = cv2.bitwise_and(
        roi_filter_img_rgb, roi_filter_img_rgb, mask=roi_filter_mask
    )
    # convert for cv
    roi_filter_img_rgb = cv2.cvtColor(roi_filter_img_rgb, cv2.COLOR_RGB2BGR)
    roi_filter_result = cv2.cvtColor(roi_filter_result, cv2.COLOR_RGB2BGR)
    if save:
        dataHandler.saveRGBImage(roi_filter_img_rgb, saveFolderPath, "roiFilter_Input")
        dataHandler.saveRGBImage(roi_filter_mask, saveFolderPath, "roiFilter_Mask")
        dataHandler.saveRGBImage(roi_filter_result, saveFolderPath, "roiFilter_Result")
    if show:
        cv2.imshow(
            "Color filter input",
            cv2.resize(
                roi_filter_img_rgb,
                None,
                fx=0.25,
                fy=0.25,
            ),
        )
        cv2.imshow(
            "Color filter mask",
            cv2.resize(
                roi_filter_mask,
                None,
                fx=0.25,
                fy=0.25,
            ),
        )
        cv2.imshow(
            "Color filter result",
            cv2.resize(
                roi_filter_result,
                None,
                fx=0.25,
                fy=0.25,
            ),
        )
        cv2.waitKey(0)
