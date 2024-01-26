import sys
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

try:
    sys.path.append(os.getcwd().replace("/plot", ""))
    from src.sensing.dataHandler import DataHandler
except:
    print("Imports for plotting image processing methods failed.")
    raise

# control parameters
save = True
show = False

relFilePath_Img = "data/darus_data_download/data/20230511_HandOcclusion/20230511_104424_Arena/data/20230511_104530_448898_image_rgb.png"
relFilePath_Background = "data/darus_data_download/data/20230511_132621_Background/data/20230511_132621_368078_image_rgb.png"
saveFolderPath = "imgs/plots_imageProcessingMethods/"


# color filter values
max_value_H = 360 // 2
low_H = 0
low_S = 100
low_V = 0
high_H = 20
high_S = 255
high_V = 255

if __name__ == "__main__":
    # load image
    dataHandler = DataHandler()

    # color filter
    color_filter_img_rgb = dataHandler.loadImage(relFilePath_Img)

    color_filter_img_hsv = cv2.cvtColor(color_filter_img_rgb, cv2.COLOR_RGB2HSV)
    color_filter_mask = cv2.inRange(
        color_filter_img_hsv, (low_H, low_S, low_V), (high_H, high_S, high_V)
    )
    color_filter_result = cv2.bitwise_and(
        color_filter_img_rgb, color_filter_img_rgb, mask=color_filter_mask
    )
    # color_filter_result[np.where((color_filter_result == [0, 0, 0]).all(axis=2))] = [
    #     255,
    #     255,
    #     255,
    # ]
    # convert for cv
    color_filter_img_rgb = cv2.cvtColor(color_filter_img_rgb, cv2.COLOR_RGB2BGR)
    color_filter_result = cv2.cvtColor(color_filter_result, cv2.COLOR_RGB2BGR)
    if save:
        dataHandler.saveRGBImage(
            color_filter_img_rgb, saveFolderPath, "colorFilter_Input"
        )
        dataHandler.saveRGBImage(color_filter_mask, saveFolderPath, "colorFilter_Mask")
        dataHandler.saveRGBImage(
            color_filter_result, saveFolderPath, "colorFilter_Result"
        )
    if show:
        cv2.imshow(
            "Color filter input",
            cv2.resize(
                color_filter_img_rgb,
                None,
                fx=0.25,
                fy=0.25,
            ),
        )
        cv2.imshow(
            "Color filter mask",
            cv2.resize(
                color_filter_mask,
                None,
                fx=0.25,
                fy=0.25,
            ),
        )
        cv2.imshow(
            "Color filter result",
            cv2.resize(
                color_filter_result,
                None,
                fx=0.25,
                fy=0.25,
            ),
        )
        cv2.waitKey(0)
