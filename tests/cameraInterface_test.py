import os, sys

try:
    sys.path.append(os.getcwd().replace("/tests", ""))
    from src.sensing.cameraInterface import CameraInterface
except:
    print("Imports for testing camera interface failed.")
    raise

savePath = "data/acquired_data/20230425_TestDataAcquisiton/"

def testCameraInteface():
    testInterface = CameraInterface()
    
    testInterface.displayLeftRBGImage()
    testInterface.streamLeftRBGImage()

if __name__ == "__main__":
    testCameraInteface()

