import os, sys
import numpy as np
try:
    sys.path.append(os.getcwd().replace("/tests", ""))
    from src.sensing.dataHandler import DataHandler
except:
    print("Imports for testing camera interface failed.")
    raise

folderPath = "data/acquiredData/20230516_184346_DataSet/"

def testLoadNumpyArrayFromBinary():
    testDataHandler = DataHandler(folderPath)
    testDisparityMap = testDataHandler.loadNumpyArrayFromBinary("20230425_163241841358_map_disparity.npy")
    print(testDisparityMap)

def testLoadRGBImage():
    testDataHandler = DataHandler(folderPath)
    testRGBImageArray = testDataHandler.loadNumpyArrayFromPNG("20230425_163241841358_image_rgb.png")
    print(testRGBImageArray)

def testLoadGrayScaleImage():
    testDataHandler = DataHandler(folderPath)
    testGrayscaleImageArray = testDataHandler.loadNumpyArrayFromPNG("20230425_163241841358_image_disparity.png", type = "grayscale")
    print(testGrayscaleImageArray)

def loadDisparityMapFromTIF():
    testDataHandler = DataHandler(folderPath)
    disparity_data_fromTIF = testDataHandler.loadDisparityMapFromTIF("20230516_184346_736858_image_disparity.tif")
    disparity_data_fromNumpy = testDataHandler.loadNumpyArrayFromBinary("20230516_184346_736858_map_disparity.npy")
    print(np.sum(disparity_data_fromTIF-disparity_data_fromNumpy))

if __name__ == "__main__":
    #testLoadNumpyArrayFromBinary()
    #testLoadRGBImage()
    #testLoadGrayScaleImage()
    loadDisparityMapFromTIF()
