import sys
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

try:
    sys.path.append(os.getcwd().replace("/experimental", ""))
    from src.sensing.dataHandler import DataHandler
except:
    print("Imports for Application determineHSVFilterValues failed.")
    raise

relFilePath_Img = "data/darus_data_download/data/20230511_HandOcclusion/20230511_104424_Arena/data/20230511_104528_526268_image_rgb.png"

relFilePath_Background = "data/darus_data_download/data/20230511_132621_Background/data/20230511_132621_368078_image_rgb.png"

if __name__ == "__main__":
    # load image
    dataHandler = DataHandler()
    img = dataHandler.loadImage(relFilePath_Img)
    background = dataHandler.loadImage(relFilePath_Background)

    inverted = img.copy()
    subtracted = img - background
    # subtracted = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) - cv2.cvtColor(
    #     background, cv2.COLOR_BGR2GRAY
    # )
    # apply threshold filter
    threshold = 30
    # img[
    #     np.abs(np.sum(img.astype(int) - background.astype(int), axis=2) / 3) < threshold
    # ] = 255
    img[
        np.abs(img.astype(int)[:, :, 0] - background.astype(int)[:, :, 0]) < threshold
    ] = 255
    img[
        np.abs(img.astype(int)[:, :, 1] - background.astype(int)[:, :, 1]) < threshold
    ] = 255
    img[
        np.abs(img.astype(int)[:, :, 2] - background.astype(int)[:, :, 2]) < threshold
    ] = 255
    inverted[
        np.invert(
            np.abs(np.sum(img.astype(int) - background.astype(int), axis=2) / 3)
            > threshold
        )
    ] = 255
    mask = cv2.threshold(
        cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), 127, 255, cv2.THRESH_BINARY_INV
    )[1]
    cv2.imshow(
        "filtered img: ",
        cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), None, fx=0.25, fy=0.25),
    )
    cv2.imshow(
        "subtracted img: ",
        cv2.resize(cv2.cvtColor(subtracted, cv2.COLOR_RGB2BGR), None, fx=0.25, fy=0.25),
    )
    cv2.imshow(
        "inverted img: ",
        cv2.resize(cv2.cvtColor(inverted, cv2.COLOR_RGB2BGR), None, fx=0.25, fy=0.25),
    )
    cv2.imshow(
        "mask img:",
        cv2.resize(mask, None, fx=0.25, fy=0.25),
    )
    cv2.waitKey(0)
