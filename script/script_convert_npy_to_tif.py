import sys, os

import numpy as np
try:
    sys.path.append(os.getcwd().replace("/tests", ""))
    from src.sensing.dataHandler import DataHandler
except:
    print("Imports for testing camera interface failed.")
    raise


def convert_npy_to_tif(dirpath, fileName):
    # load stereodata
    dataSetFolderPath = "/".join(dirpath.split("/")[:-1])
    dataAcqusition = DataHandler(dataSetFolderPath)
    dataAcqusition.defaultLoadFolderPath = dataSetFolderPath + "/"
    dataAcqusition.defaultLoadFolderPath_Data = dataSetFolderPath +"/data/"
    cameraParameters = dataAcqusition.loadCameraParameters("cameraParameters.json")
    #load disparity map
    disparityMap = dataAcqusition.loadNumpyArrayFromBinary(fileName)
    # convert to integer array
    disparityData = (disparityMap * cameraParameters["disparityRangeFactor"]).astype(np.uint16)
    # save as tif
    tifFileName = "_".join(fileName.split("_")[:3]) + "_image_disparity"
    dataAcqusition.saveDisparityImage(disparityData,dataAcqusition.defaultLoadFolderPath_Data, tifFileName)
    # test
    test_disparityMap = dataAcqusition.loadDisparityMapFromTIF(tifFileName+".tif")
    testResult = np.sum(test_disparityMap - disparityMap)
    print(testResult)
    return
    
if __name__ == "__main__":
    rootDirectory = "data/acquiredData/data_convert_npy_to_tif"
    relevantDirectories = []
    for dirpath, dirnames, filenames in os.walk(rootDirectory):
        for filename in [f for f in filenames if f.endswith(".npy")]:
            print(os.path.join(dirpath, filename))
            try:
                convert_npy_to_tif(dirpath, filename)
            except:
                print("Could not convert files in directory {}".format(dirpath))
