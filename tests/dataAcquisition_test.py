import os, sys

try:
    sys.path.append(os.getcwd().replace("/tests", ""))
    from src.sensing.cameraInterface import CameraInterface
    from src.sensing.dataAcquisition import DataAcquisition
except:
    print("Imports for testing camera interface failed.")
    raise

savePath = "data/acquired_data/20230425_TestDataAcquisiton/"

def testDataAcquisition():

    testAcquisition = DataAcquisition(savePath)
    testAcquisition.recordStereoDataSet()

def testRecordStereoDataSetsManually():
    testAcquisition = DataAcquisition(savePath)
    testAcquisition.recordStereoDataSets(method="manual")

def testRecordStereoDataSetsAutomatically():
    testAcquisition = DataAcquisition(savePath)
    testAcquisition.recordStereoDataSets(method="auto")

if __name__ == "__main__":
    testDataAcquisition()
    #testRecordStereoDataSetsManually()
    #testRecordStereoDataSetsAutomatically()