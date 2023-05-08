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

folderPath = (
    "data/darus_data_download/data/20230505_arenawireharness_manipulationsequence/"
)
fileName = "20230505_174651469438_image_rgb.png"

# load image
dataHandler = DataHandler(folderPath)
rgbImage = dataHandler.loadNumpyArrayFromPNG(fileName)

imageHeight = rgbImage.shape[0]
imageWidth = rgbImage.shape[1]

max_value = 255
max_value_H = 360 // 2
leftBoarder = 0
lowerBoarder = 0
rightBoarder = imageWidth
upperBoarder = imageHeight
image_window__name = "Image"
mask_window_name = "Mask"
diff_window_name = "Difference"
left_boarder_name = "uMin"
lower_boarder_name = "vMin"  # lower and upper relative to image coordinates (inverted to how user perceives the image)
right_boarder_name = "uMax"
upper_boarder_name = "vMax"


def on_left_boarder_thresh_trackbar(val):
    global leftBoarder
    global rightBoarder
    leftBoarder = val
    leftBoarder = min(rightBoarder - 1, leftBoarder)
    cv2.setTrackbarPos(left_boarder_name, mask_window_name, leftBoarder)


def on_right_boarder_thresh_trackbar(val):
    global leftBoarder
    global rightBoarder
    rightBoarder = val
    rightBoarder = max(rightBoarder, leftBoarder + 1)
    cv2.setTrackbarPos(right_boarder_name, image_window__name, rightBoarder)


def on_lower_boarder_thresh_trackbar(val):
    global lowerBoarder
    global upperBoarder
    lowerBoarder = val
    lowerBoarder = min(upperBoarder - 1, lowerBoarder)
    cv2.setTrackbarPos(lower_boarder_name, mask_window_name, lowerBoarder)


def on_upper_boarder_thresh_trackbar(val):
    global lowerBoarder
    global upperBoarder
    upperBoarder = val
    upperBoarder = max(upperBoarder, lowerBoarder + 1)
    cv2.setTrackbarPos(upper_boarder_name, mask_window_name, upperBoarder)


if __name__ == "__main__":
    # create windows
    cv2.namedWindow(image_window__name)
    cv2.namedWindow(mask_window_name)
    cv2.namedWindow(diff_window_name)
    cv2.createTrackbar(
        left_boarder_name,
        mask_window_name,
        leftBoarder,
        imageWidth,
        on_left_boarder_thresh_trackbar,
    )
    cv2.createTrackbar(
        right_boarder_name,
        mask_window_name,
        rightBoarder,
        imageWidth,
        on_right_boarder_thresh_trackbar,
    )
    cv2.createTrackbar(
        lower_boarder_name,
        mask_window_name,
        lowerBoarder,
        imageHeight,
        on_lower_boarder_thresh_trackbar,
    )
    cv2.createTrackbar(
        upper_boarder_name,
        mask_window_name,
        upperBoarder,
        imageHeight,
        on_upper_boarder_thresh_trackbar,
    )
    while True:
        mask = np.zeros((imageHeight, imageWidth), dtype=np.uint8)
        mask[lowerBoarder:upperBoarder, leftBoarder:rightBoarder] = 255
        frame_diff = cv2.bitwise_and(rgbImage, rgbImage, mask=mask)
        frame_diff[np.where((frame_diff == [0, 0, 0]).all(axis=2))] = [
            255,
            255,
            255,
        ]  # invert such that filtered values are displayed white
        cv2.imshow(image_window__name, cv2.resize(rgbImage, None, fx=0.25, fy=0.25))
        cv2.imshow(mask_window_name, cv2.resize(mask, None, fx=0.25, fy=0.25))
        cv2.imshow(diff_window_name, cv2.resize(frame_diff, None, fx=0.25, fy=0.25))
        key = cv2.waitKey(30)
        if key == ord("q") or key == 27:
            break
