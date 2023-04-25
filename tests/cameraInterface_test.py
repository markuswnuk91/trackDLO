import os, sys

try:
    sys.path.append(os.getcwd().replace("/tests", ""))
    from src.sensing.cameraInterface import CameraInterface
    from src.sensing.dataAcquisition import DataAcquisition
except:
    print("Imports for testing camera interface failed.")
    raise


def testCameraInteface():
    testInterface = CameraInterface()
    
    testInterface.displayLeftRBGImage()
    
    testInterface.streamLeftRBGImage()


def testDataAcquisition():
    savePath = "data/sensor_data/20230425_TestDataAcquisiton/"
    testAcquisition = DataAcquisition(savePath)
    testAcquisition.acquireStereoDataSet()

if __name__ == "__main__":
    #testCameraInteface()
    testDataAcquisition()
