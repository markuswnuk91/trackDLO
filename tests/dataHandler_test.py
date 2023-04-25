import os, sys

try:
    sys.path.append(os.getcwd().replace("/tests", ""))
    from src.sensing.dataHandler import DataHandler
except:
    print("Imports for testing camera interface failed.")
    raise

folderPath = "data/acquired_data/20230425_TestDataAcquisiton/"

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

if __name__ == "__main__":
    #testLoadNumpyArrayFromBinary()
    testLoadRGBImage()
    testLoadGrayScaleImage()
