import os, sys

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
    pointCloud = pointCloudProcessor.calculatePointCloudFromImageAndDisparity(rgbImage,disparityMap, cameraParameter["qmatrix"])
    print(pointCloud)

if __name__ == "__main__":
    testPointCloudGeneration()