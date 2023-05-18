import sys, os
import os.path
import numpy as np
try:
    sys.path.append(os.getcwd().replace("/tests", ""))
    from src.sensing.dataHandler import DataHandler
except:
    print("Imports for testing camera interface failed.")
    raise


def compress_tif(dirpath, fileName):
    # load stereodata
    dataSetFolderPath = "/".join(dirpath.split("/")[:-1])
    dataAcqusition = DataHandler(dataSetFolderPath)
    dataAcqusition.defaultLoadFolderPath = dataSetFolderPath + "/"
    dataAcqusition.defaultLoadFolderPath_Data = dataSetFolderPath +"/data/"
    cameraParameters = dataAcqusition.loadCameraParameters("cameraParameters.json")
    #load disparity information
    disparityMap = dataAcqusition.loadDisparityDataFromTIF(fileName)
    # convert to integer array
    disparityData = (disparityMap * cameraParameters["disparityRangeFactor"]).astype(np.uint16)
    # save as compressed tif
    tifFileName = "_".join(fileName.split("_")[:3]) + "_image_disparity"
    dataAcqusition.saveDisparityImage(disparityData,dataAcqusition.defaultLoadFolderPath_Data, tifFileName)
    # test
    test_disparityMap = dataAcqusition.loadDisparityMapFromTIF(tifFileName+".tif")
    testResult = np.sum(test_disparityMap - disparityMap)
    print(testResult)
    return
    
if __name__ == "__main__":
    # find all folders that contain tif files to be compressed
    rootDirectory = "data/acquiredData"
    relevantDirectories = []
    for dirpath, dirnames, filenames in os.walk(rootDirectory):
        for filename in [f for f in filenames if f.endswith(".tif")]:
            print(os.path.join(dirpath, filename))
            try:
                compress_tif(dirpath, filename)
            except:
                print("Could not convert {}".format(filename))
    # for dataSetFolderPath in dataSetFolderPaths:
    #     compress_tif(folderPath)