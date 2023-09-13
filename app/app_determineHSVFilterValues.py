import sys
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

try:
    sys.path.append(os.getcwd().replace("/app", ""))
    from src.sensing.dataHandler import DataHandler
except:
    print("Imports for Application determineHSVFilterValues failed.")
    raise

relFilePath = "data/darus_data_download/data/202230603_Configurations_mounted/20230603_143937_modelY/data/20230603_144611_483120_image_rgb.png"

fileName = os.path.basename(relFilePath)
dataSetFolderPath = os.path.dirname(os.path.dirname(relFilePath)) + "/"

max_value = 255
max_value_H = 360 // 2
low_H = 0
low_S = 0
low_V = 0
high_H = max_value_H
high_S = max_value
high_V = max_value
image_window__name = "Image"
mask_window_name = "Mask"
diff_window_name = "Difference"
low_H_name = "Low H"
low_S_name = "Low S"
low_V_name = "Low V"
high_H_name = "High H"
high_S_name = "High S"
high_V_name = "High V"


def on_low_H_thresh_trackbar(val):
    global low_H
    global high_H
    low_H = val
    low_H = min(high_H - 1, low_H)
    cv2.setTrackbarPos(low_H_name, mask_window_name, low_H)


def on_high_H_thresh_trackbar(val):
    global low_H
    global high_H
    high_H = val
    high_H = max(high_H, low_H + 1)
    cv2.setTrackbarPos(high_H_name, image_window__name, high_H)


def on_low_S_thresh_trackbar(val):
    global low_S
    global high_S
    low_S = val
    low_S = min(high_S - 1, low_S)
    cv2.setTrackbarPos(low_S_name, mask_window_name, low_S)


def on_high_S_thresh_trackbar(val):
    global low_S
    global high_S
    high_S = val
    high_S = max(high_S, low_S + 1)
    cv2.setTrackbarPos(high_S_name, mask_window_name, high_S)


def on_low_V_thresh_trackbar(val):
    global low_V
    global high_V
    low_V = val
    low_V = min(high_V - 1, low_V)
    cv2.setTrackbarPos(low_V_name, mask_window_name, low_V)


def on_high_V_thresh_trackbar(val):
    global low_V
    global high_V
    high_V = val
    high_V = max(high_V, low_V + 1)
    cv2.setTrackbarPos(high_V_name, mask_window_name, high_V)


def filterImageHSV(hMin, hMax, sMin, sMax, vMin, vMax):
    hsvImage = cv2.cvtColor(rgbImage, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsvImage, (hMin, sMin, vMin), (hMax, sMax, vMax))


if __name__ == "__main__":
    # load image
    dataHandler = DataHandler(dataSetFolderPath)
    rgbImage = dataHandler.loadNumpyArrayFromPNG(fileName)

    # create windows
    cv2.namedWindow(image_window__name)
    cv2.namedWindow(mask_window_name)
    cv2.namedWindow(diff_window_name)
    cv2.createTrackbar(
        low_H_name, mask_window_name, low_H, max_value_H, on_low_H_thresh_trackbar
    )
    cv2.createTrackbar(
        high_H_name, mask_window_name, high_H, max_value_H, on_high_H_thresh_trackbar
    )
    cv2.createTrackbar(
        low_S_name, mask_window_name, low_S, max_value, on_low_S_thresh_trackbar
    )
    cv2.createTrackbar(
        high_S_name, mask_window_name, high_S, max_value, on_high_S_thresh_trackbar
    )
    cv2.createTrackbar(
        low_V_name, mask_window_name, low_V, max_value, on_low_V_thresh_trackbar
    )
    cv2.createTrackbar(
        high_V_name, mask_window_name, high_V, max_value, on_high_V_thresh_trackbar
    )
    while True:
        # mask = filterImageHSV()
        frame_HSV = cv2.cvtColor(rgbImage, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))
        # mask_inv = cv2.bitwise_not(mask)
        frame_diff = cv2.bitwise_and(rgbImage, rgbImage, mask=mask)
        frame_diff[np.where((frame_diff == [0, 0, 0]).all(axis=2))] = [255, 255, 255]
        cv2.imshow(
            image_window__name,
            cv2.resize(
                cv2.cvtColor(rgbImage, cv2.COLOR_RGB2BGR), None, fx=0.25, fy=0.25
            ),
        )
        cv2.imshow(mask_window_name, cv2.resize(mask, None, fx=0.25, fy=0.25))
        cv2.imshow(
            diff_window_name,
            cv2.resize(
                cv2.cvtColor(frame_diff, cv2.COLOR_RGB2BGR), None, fx=0.25, fy=0.25
            ),
        )
        key = cv2.waitKey(30)
        if key == ord("q") or key == 27:
            break
