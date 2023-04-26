import os, sys
import matplotlib.pyplot as plt
try:
    sys.path.append(os.getcwd().replace("/tests", ""))
    from src.sensing.pointCloudProcessing import PointCloudProcessing
    from src.visualization.plot3D import *
except:
    print("Imports for testing camera interface failed.")
    raise

folderPath = "data/acquired_data/20230426_133624_DataSet/"
fileName_rgb = "20230426_133625761810_image_rgb.png"
fileName_disp = "20230426_133625761810_map_disparity.npy"
vis = True

def visualizePointCloud(points,colors):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=colors,
        s=1,
        alpha = 0.1,
        )
    # set axis limits
    # ax.set_xlim(-1, 1)
    # ax.set_ylim(-1, 1)
    # ax.set_zlim(0, 1.5)
    return ax

def visualizeBoundingBox(ax, xMin, xMax, yMin, yMax, zMin, zMax):
    plotCube(ax, xMin, xMax, yMin, yMax, zMin, zMax)

def testPointCloudGeneration():
    pointCloudProcessor = PointCloudProcessing(folderPath)
    rgbImage = pointCloudProcessor.loadNumpyArrayFromPNG(fileName_rgb)
    disparityMap = pointCloudProcessor.loadNumpyArrayFromBinary(fileName_disp)
    cameraParameter = pointCloudProcessor.loadCameraParameters("cameraParameters.json")
    pointCloud = pointCloudProcessor.calculatePointCloud(rgbImage,disparityMap, cameraParameter["qmatrix"])

    xyzPoints = pointCloud[0]
    colors = pointCloud[1] / 255

    #downsample
    xyzPoints = xyzPoints[::10,:]
    colors = colors[::10,:]
    if vis:
        visualizePointCloud(xyzPoints, colors)
        plt.show(block=True)
    return pointCloud

def testPointCloudFilter():
    pointCloudProcessor = PointCloudProcessing(folderPath)
    rgbImage = pointCloudProcessor.loadNumpyArrayFromPNG(fileName_rgb)
    disparityMap = pointCloudProcessor.loadNumpyArrayFromBinary(fileName_disp)
    cameraParameter = pointCloudProcessor.loadCameraParameters("cameraParameters.json")
    pointCloud = pointCloudProcessor.calculatePointCloud(rgbImage,disparityMap, cameraParameter["qmatrix"])

    xyzPoints = pointCloud[0]
    colors = pointCloud[1] / 255

    #downsampling
    xyzPoints = xyzPoints[::100,:]
    colors = colors[::100,:]

    # box filtering
    mask = pointCloudProcessor.getMaskFromPointCloud_BoundingBox(xyzPoints, -1, 1, -1, 1, 0, 1.5)
    xyzPoints = xyzPoints[mask,:]
    colors = colors[mask,:]
    if vis:
        ax = visualizePointCloud(xyzPoints, colors)
        visualizeBoundingBox(ax, 0.15, 0.35, -0.2, 0, 0.9, 1.1)
        plt.show(block=True)
    return pointCloud

if __name__ == "__main__":
    # testPointCloudGeneration()
    testPointCloudFilter()