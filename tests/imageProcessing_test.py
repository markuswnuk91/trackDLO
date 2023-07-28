import sys
import os
import matplotlib.pyplot as plt
import cv2

try:
    sys.path.append(os.getcwd().replace("/app", ""))
    from src.sensing.imageProcessing import ImageProcessing
except:
    print("Imports for testing image processing class failed.")
    raise

relFilePath = "data/darus_data_download/data/202230603_Configurations_mounted/20230603_143937_modelY/data/20230603_144220_681561_image_rgb.png"

# scipt starts here
folderPath = os.path.dirname(os.path.dirname(relFilePath)) + "/"
fileName = os.path.basename(relFilePath)


def getParametersFilterRed():
    hueMin = 0
    hueMax = 15
    saturationMin = 120
    saturationMax = 255
    valueMin = 165
    valueMax = 255
    return hueMin, hueMax, saturationMin, saturationMax, valueMin, valueMax


def getParametersFilterGreen():
    hueMin = 53
    hueMax = 76
    saturationMin = 35
    saturationMax = 164
    valueMin = 52
    valueMax = 255
    return hueMin, hueMax, saturationMin, saturationMax, valueMin, valueMax


def getParametersFilterBlue():
    hueMin = 118
    hueMax = 160
    saturationMin = 110
    saturationMax = 255
    valueMin = 0
    valueMax = 255
    return hueMin, hueMax, saturationMin, saturationMax, valueMin, valueMax


def getParametersFilterBright():
    hueMin = 0
    hueMax = 180
    saturationMin = 0
    saturationMax = 255
    valueMin = 100
    valueMax = 255
    return hueMin, hueMax, saturationMin, saturationMax, valueMin, valueMax


def getParametersFilterDark():
    hueMin = 0
    hueMax = 180
    saturationMin = 0
    saturationMax = 255
    valueMin = 0
    valueMax = 85
    return hueMin, hueMax, saturationMin, saturationMax, valueMin, valueMax


def testHSVFilter():
    original_image_window_name = "Original Image"

    mask_red_window_name = "Mask Red"
    mask_green_window_name = "Mask Green"
    mask_blue_window_name = "Mask Blue"
    mask_bright_window_name = "Mask Bright"
    mask_dark_window_name = "Mask Dark"

    filtered_red_window_name = "Filtered Red"
    filtered_green_window_name = "Filtered Green"
    filtered_blue_window_name = "Filtered Blue"
    filtered_bright_window_name = "Filtered Bright"
    filtered_dark_window_name = "Filtered Dark"

    imageProcessor = ImageProcessing(folderPath)
    rgbImage = imageProcessor.loadNumpyArrayFromPNG(fileName)

    cv2.namedWindow(original_image_window_name)

    # red
    cv2.namedWindow(mask_red_window_name)
    cv2.namedWindow(filtered_red_window_name)
    # green
    cv2.namedWindow(mask_green_window_name)
    cv2.namedWindow(filtered_green_window_name)
    # blue
    cv2.namedWindow(mask_blue_window_name)
    cv2.namedWindow(filtered_blue_window_name)
    # bright
    cv2.namedWindow(mask_bright_window_name)
    cv2.namedWindow(filtered_bright_window_name)
    # dark
    cv2.namedWindow(mask_dark_window_name)
    cv2.namedWindow(filtered_dark_window_name)

    # filter red
    (
        hueMin,
        hueMax,
        saturationMin,
        saturationMax,
        valueMin,
        valueMax,
    ) = getParametersFilterRed()
    maskFilter_Red = imageProcessor.getMaskFromRGB_applyHSVFilter(
        rgbImage, hueMin, hueMax, saturationMin, saturationMax, valueMin, valueMax
    )
    fileredImage_Red = imageProcessor.filterRGB_applyHSVFilter(
        rgbImage, hueMin, hueMax, saturationMin, saturationMax, valueMin, valueMax
    )

    # filter green
    (
        hueMin,
        hueMax,
        saturationMin,
        saturationMax,
        valueMin,
        valueMax,
    ) = getParametersFilterGreen()
    maskFilter_Green = imageProcessor.getMaskFromRGB_applyHSVFilter(
        rgbImage, hueMin, hueMax, saturationMin, saturationMax, valueMin, valueMax
    )
    fileredImage_Green = imageProcessor.filterRGB_applyHSVFilter(
        rgbImage, hueMin, hueMax, saturationMin, saturationMax, valueMin, valueMax
    )

    # filter blue
    (
        hueMin,
        hueMax,
        saturationMin,
        saturationMax,
        valueMin,
        valueMax,
    ) = getParametersFilterBlue()
    maskFilter_Blue = imageProcessor.getMaskFromRGB_applyHSVFilter(
        rgbImage, hueMin, hueMax, saturationMin, saturationMax, valueMin, valueMax
    )
    fileredImage_Blue = imageProcessor.filterRGB_applyHSVFilter(
        rgbImage, hueMin, hueMax, saturationMin, saturationMax, valueMin, valueMax
    )

    # filter bright
    (
        hueMin,
        hueMax,
        saturationMin,
        saturationMax,
        valueMin,
        valueMax,
    ) = getParametersFilterBright()
    maskFilter_Bright = imageProcessor.getMaskFromRGB_applyHSVFilter(
        rgbImage, hueMin, hueMax, saturationMin, saturationMax, valueMin, valueMax
    )
    fileredImage_Bright = imageProcessor.filterRGB_applyHSVFilter(
        rgbImage, hueMin, hueMax, saturationMin, saturationMax, valueMin, valueMax
    )
    # filter dark
    (
        hueMin,
        hueMax,
        saturationMin,
        saturationMax,
        valueMin,
        valueMax,
    ) = getParametersFilterDark()
    maskFilter_Dark = imageProcessor.getMaskFromRGB_applyHSVFilter(
        rgbImage, hueMin, hueMax, saturationMin, saturationMax, valueMin, valueMax
    )
    fileredImage_Dark = imageProcessor.filterRGB_applyHSVFilter(
        rgbImage, hueMin, hueMax, saturationMin, saturationMax, valueMin, valueMax
    )
    # show images
    cv2.imshow(
        original_image_window_name,
        cv2.resize(cv2.cvtColor(rgbImage, cv2.COLOR_RGB2BGR), None, fx=0.25, fy=0.25),
    )
    # red
    cv2.imshow(mask_red_window_name, cv2.resize(maskFilter_Red, None, fx=0.25, fy=0.25))
    cv2.imshow(
        filtered_red_window_name,
        cv2.resize(
            cv2.cvtColor(fileredImage_Red, cv2.COLOR_RGB2BGR), None, fx=0.25, fy=0.25
        ),
    )
    # green
    cv2.imshow(
        mask_green_window_name, cv2.resize(maskFilter_Green, None, fx=0.25, fy=0.25)
    )
    cv2.imshow(
        filtered_green_window_name,
        cv2.resize(
            cv2.cvtColor(fileredImage_Green, cv2.COLOR_RGB2BGR), None, fx=0.25, fy=0.25
        ),
    )
    # blue
    cv2.imshow(
        mask_blue_window_name, cv2.resize(maskFilter_Blue, None, fx=0.25, fy=0.25)
    )
    cv2.imshow(
        filtered_blue_window_name,
        cv2.resize(
            cv2.cvtColor(fileredImage_Blue, cv2.COLOR_RGB2BGR), None, fx=0.25, fy=0.25
        ),
    )
    # bright
    cv2.imshow(
        mask_bright_window_name, cv2.resize(maskFilter_Bright, None, fx=0.25, fy=0.25)
    )
    cv2.imshow(
        filtered_bright_window_name,
        cv2.resize(
            cv2.cvtColor(fileredImage_Bright, cv2.COLOR_RGB2BGR), None, fx=0.25, fy=0.25
        ),
    )
    # dark
    cv2.imshow(
        mask_dark_window_name, cv2.resize(maskFilter_Dark, None, fx=0.25, fy=0.25)
    )
    cv2.imshow(
        filtered_dark_window_name,
        cv2.resize(
            cv2.cvtColor(fileredImage_Dark, cv2.COLOR_RGB2BGR), None, fx=0.25, fy=0.25
        ),
    )
    key = cv2.waitKey(0)
    return


def testROIFilter():
    original_image_window_name = "Original Image"
    mask_roi_window_name = "Mask ROI"
    filtered_window_name = "Image ROI"
    imageProcessor = ImageProcessing(folderPath)
    rgbImage = imageProcessor.loadNumpyArrayFromPNG(fileName)
    width = rgbImage.shape[1]
    height = rgbImage.shape[0]
    uMin = int(1 / 5 * width)
    uMax = int((1 - 1 / 5) * width)
    vMin = int(1 / 5 * height)
    vMax = int((1 - 1 / 5) * height)
    maskFilter_ROI = imageProcessor.getMaskFromRGB_applyROI(
        rgbImage, uMin, uMax, vMin, vMax
    )
    fileredImage_ROI = imageProcessor.filterRGB_applyROI(
        rgbImage, uMin, uMax, vMin, vMax
    )
    cv2.namedWindow(original_image_window_name)
    cv2.namedWindow(mask_roi_window_name)
    cv2.namedWindow(filtered_window_name)
    # show
    cv2.imshow(
        original_image_window_name,
        cv2.resize(cv2.cvtColor(rgbImage, cv2.COLOR_RGB2BGR), None, fx=0.25, fy=0.25),
    )
    cv2.imshow(mask_roi_window_name, cv2.resize(maskFilter_ROI, None, fx=0.25, fy=0.25))
    cv2.imshow(
        filtered_window_name,
        cv2.resize(
            cv2.cvtColor(fileredImage_ROI, cv2.COLOR_RGB2BGR), None, fx=0.25, fy=0.25
        ),
    )
    key = cv2.waitKey(0)
    return


if __name__ == "__main__":
    # testHSVFilter()
    testROIFilter()
