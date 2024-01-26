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
threshold = 30

relFilePath_Img = "data/darus_data_download/data/20230511_HandOcclusion/20230511_104424_Arena/data/20230511_104530_448898_image_rgb.png"
relFilePath_Background = "data/darus_data_download/data/20230511_132621_Background/data/20230511_132621_368078_image_rgb.png"
saveFolderPath = "imgs/plots_imageProcessingMethods/"

if __name__ == "__main__":
    dataHandler = DataHandler()
    imageProcessor = ImageProcessing()

    # load image
    background_filter_img_rgb = dataHandler.loadImage(relFilePath_Img)
    background = dataHandler.loadImage(relFilePath_Background)
    # apply filter

    background_filter_result = background_filter_img_rgb.copy()
    # background_filter_result[
    #     np.abs(
    #         background_filter_result.astype(int)[:, :, 0]
    #         - background.astype(int)[:, :, 0]
    #     )
    #     < threshold
    # ] = 255
    # background_filter_result[
    #     np.abs(
    #         background_filter_result.astype(int)[:, :, 1]
    #         - background.astype(int)[:, :, 1]
    #     )
    #     < threshold
    # ] = 255
    # background_filter_result[
    #     np.abs(
    #         background_filter_result.astype(int)[:, :, 2]
    #         - background.astype(int)[:, :, 2]
    #     )
    #     < threshold
    # ] = 255
    # background_filter_mask = cv2.threshold(
    #     cv2.cvtColor(background_filter_result, cv2.COLOR_RGB2GRAY),
    #     threshold,
    #     255,
    #     cv2.THRESH_BINARY_INV,
    # )[1]
    boolean_array = np.logical_and(
        np.abs(
            background_filter_result.astype(int)[:, :, 0]
            - background.astype(int)[:, :, 0]
        )
        < threshold,
        np.abs(
            background_filter_result.astype(int)[:, :, 1]
            - background.astype(int)[:, :, 1]
        )
        < threshold,
        np.abs(
            background_filter_result.astype(int)[:, :, 2]
            - background.astype(int)[:, :, 2]
        )
        < threshold,
    )
    background_filter_result[boolean_array] = 0
    background_filter_mask = np.ones((background_filter_result.shape)) * 255
    background_filter_mask[boolean_array] = 0
    # convert for cv
    background_filter_img_rgb = cv2.cvtColor(
        background_filter_img_rgb, cv2.COLOR_RGB2BGR
    )
    background_filter_result = cv2.cvtColor(background_filter_result, cv2.COLOR_RGB2BGR)
    background = cv2.cvtColor(background, cv2.COLOR_RGB2BGR)
    if save:
        dataHandler.saveRGBImage(
            background_filter_img_rgb, saveFolderPath, "backgroundSubtraction_Input"
        )
        dataHandler.saveRGBImage(
            background, saveFolderPath, "backgroundSubtraction_Background"
        )
        dataHandler.saveRGBImage(
            background_filter_mask, saveFolderPath, "backgroundSubtraction_Mask"
        )
        dataHandler.saveRGBImage(
            background_filter_result, saveFolderPath, "backgroundSubtraction_Result"
        )
    if show:
        cv2.imshow(
            "Color filter input",
            cv2.resize(
                background_filter_img_rgb,
                None,
                fx=0.25,
                fy=0.25,
            ),
        )
        cv2.imshow(
            "Color filter mask",
            cv2.resize(
                background_filter_mask,
                None,
                fx=0.25,
                fy=0.25,
            ),
        )
        cv2.imshow(
            "Color filter result",
            cv2.resize(
                background_filter_result,
                None,
                fx=0.25,
                fy=0.25,
            ),
        )
        cv2.waitKey(0)
