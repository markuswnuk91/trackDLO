import os, sys
import matplotlib.pyplot as plt
try:
    sys.path.append(os.getcwd().replace("/tests", ""))
    from src.sensing.dataHandler import DataHandler
    from src.sensing.pointCloudProcessing import PointCloudProcessing
except:
    print("Imports for testing camera interface failed.")
    raise

folderPath = "data/acquired_data/20230425_TestDataAcquisiton/"

def testPointCloudGeneration():
    pointCloudProcessor = PointCloudProcessing(folderPath)
    rgbImage = pointCloudProcessor.loadNumpyArrayFromPNG("20230425_163241841358_image_rgb.png")
    disparityMap = pointCloudProcessor.loadNumpyArrayFromBinary("20230425_163241841358_map_disparity.npy")
    cameraParameter = pointCloudProcessor.loadCameraParameters("cameraParameters.json")
    pointCloud = pointCloudProcessor.calculatePointCloud(rgbImage,disparityMap, cameraParameter["qmatrix"])

    xyzPoints = pointCloud[0]
    colors = pointCloud[1] / 255

    #downsample
    xyzPoints = xyzPoints[::10,:]
    colors = colors[::10,:]

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(
        xyzPoints[:, 0],
        xyzPoints[:, 1],
        xyzPoints[:, 2],
        c=colors,
        s=1
        )
    # set axis limits
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(0, 1.5)
    plt.show(block=True)
if __name__ == "__main__":
    testPointCloudGeneration()