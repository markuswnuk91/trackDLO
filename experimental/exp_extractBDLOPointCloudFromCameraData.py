import sys
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
try:
    sys.path.append(os.getcwd().replace("/app", ""))
    from src.sensing.imageProcessing import ImageProcessing
    from src.sensing.pointCloudProcessing import PointCloudProcessing
    from src.visualization.plot3D import *
except:
    print("Imports for testing image processing class failed.")
    raise

visImage = True
visPointCloud = True
folderPath = "data/acquiredData/20230516_Configurations_labeled/20230516_115857_arena/"
fileName = "20230508_174803124630_image_rgb.png"


def getParametersFilterColor():
    hueMin= 53
    hueMax = 76
    saturationMin = 35
    saturationMax = 164
    valueMin = 52
    valueMax = 255
    return hueMin, hueMax, saturationMin, saturationMax, valueMin, valueMax

def getParametersFilterROI():
    uMin = 0.5
    uMax = 0.95
    vMin = 0.03
    vMax = 0.87
    return uMin, uMax, vMin, vMax

def getParametersFilterBoxOuter():
    xMin = 0
    xMax = 1
    yMin = 0
    yMax = 1
    zMin = 0
    zMax = 1
    return xMin, xMax, yMin, yMax, zMin, zMax

def getParametersFilterBoxInner():
    xMin =  0.0
    xMax = 0.2
    yMin = 0.1
    yMax = 0.3
    zMin = 0.6
    zMax = 0.8
    return xMin, xMax, yMin, yMax, zMin, zMax

def visualizePointCloud(ax, points,colors):
    plotPointSet(ax = ax, X=points, color = colors, size=1)
    return

def visualizeBoundingBox(ax, xMin, xMax, yMin, yMax, zMin, zMax):
    plotCube(ax, xMin, xMax, yMin, yMax, zMin, zMax)

def extractPointCloudFromImage():
    #imageProcessing
    imageProcessor = ImageProcessing(folderPath)
    rgbImage = imageProcessor.loadNumpyArrayFromPNG(fileName)
    # Color Filter
    (hueMin, hueMax, saturationMin, saturationMax, valueMin, valueMax) = getParametersFilterColor()
    maskFilter_Color = imageProcessor.getMaskFromRGB_applyHSVFilter(rgbImage, hueMin, hueMax, saturationMin, saturationMax, valueMin, valueMax)
    fileredImage_Color = imageProcessor.filterRGB_applyHSVFilter(rgbImage, hueMin, hueMax, saturationMin, saturationMax, valueMin, valueMax)
    # ROI Filter
    (uMin, uMax, vMin,vMax) = getParametersFilterROI()
    maskFilter_ROI = imageProcessor.getMaskFromRGB_applyROI(fileredImage_Color, uMin, uMax, vMin,vMax)
    #combine masks
    masks = [maskFilter_Color, maskFilter_ROI]
    (h, w) = maskFilter_Color.shape
    resultingMask = 255*np.ones((h,w), dtype=np.uint8)
    for mask in masks:
        resultingMask = cv2.bitwise_and(resultingMask, mask)
    maskedImage = imageProcessor.filterRGB_applyMask(rgbImage, resultingMask)
    #show 
    if visImage == True:
        cv2.imshow("Image",cv2.resize(cv2.cvtColor(rgbImage, cv2.COLOR_RGB2BGR), None, fx=.25, fy=.25))
        cv2.imshow("Combined Masks", cv2.resize(resultingMask, None, fx=.25, fy=.25))
        cv2.imshow("Masked Image", cv2.resize(cv2.cvtColor(maskedImage, cv2.COLOR_RGB2BGR), None, fx=.25, fy=.25))
        key = cv2.waitKey(0)

    #pointCloudProcessing
    pointCloudProcessor = PointCloudProcessing(folderPath)
    cameraParameter = pointCloudProcessor.loadCameraParameters("cameraParameters.json")
    disparityMap = pointCloudProcessor.loadNumpyArrayFromBinary(fileName_disp)
    points, colors = pointCloudProcessor.calculatePointCloud(rgbImage,disparityMap,cameraParameter["qmatrix"],resultingMask)
    #downsampling
    points = points[::100,:]
    colors = colors[::100,:]
    
    # BoxFilter for working area
    (xMin, xMax, yMin, yMax, zMin, zMax) = getParametersFilterBoxOuter()
    indexMaskOuter = pointCloudProcessor.getMaskFromBoundingBox(points, xMin, xMax, yMin, yMax, zMin, zMax)
    points = points[indexMaskOuter,:]
    colors = colors[indexMaskOuter,:]
    
    # Box filter for region of interest
    (xMin, xMax, yMin, yMax, zMin, zMax) = getParametersFilterBoxInner()
    indexMaskInner = pointCloudProcessor.getMaskFromBoundingBox(points, xMin, xMax, yMin, yMax, zMin, zMax)
    indexMaskOuter = np.invert(indexMaskInner)
    inliers = points[indexMaskInner,:]
    inlierColors = colors[indexMaskInner,:]
    outliers = points[indexMaskOuter,:]
    outlierColors = colors[indexMaskOuter,:]
    outlierColors[:,:] = np.array([0.5,0.5,0.5])
    points[indexMaskInner,:]
    # Visualization
    if visPointCloud:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        visualizePointCloud(ax, inliers, inlierColors)
        visualizePointCloud(ax, outliers, outlierColors)
        visualizeBoundingBox(ax, xMin, xMax, yMin, yMax, zMin, zMax)
        plt.show(block=True)

if __name__ == "__main__":
    #testHSVFilter()
    # testROIFilter()
    extractPointCloudFromImage()