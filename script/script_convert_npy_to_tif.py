import sys, os

import numpy as np
try:
    sys.path.append(os.getcwd().replace("/tests", ""))
    from src.sensing.dataAcquisition import DataAcquisition
except:
    print("Imports for testing camera interface failed.")
    raise


def convert_npy_to_tif(folderPath):
    # load stereodata
    dataAcqusition = DataAcquisition(folderPath)
    dataAcqusition.defaultLoadFolderPath = folderPath
    dataAcqusition.defaultLoadFolderPath_Data = folderPath +"data/"
    npyFiles = dataAcqusition.getDataSetFileNames_NPY()
    cameraParameters = dataAcqusition.loadCameraParameters("cameraParameters.json")
    for fileName in npyFiles:
        #load disparity map
        disparityMap = dataAcqusition.loadNumpyArrayFromBinary(fileName)
        # convert to integer array
        disparityData = disparityMap * cameraParameters["disparityRangeFactor"]
        # save as tif
        tifFileName = "_".join(fileName.split("_")[:3]) + "_image_disparity"
        #dataAcqusition.saveDisparityImage(disparityData,dataAcqusition.defaultLoadFolderPath_Data, tifFileName)
        # test
        test_disparityMap = dataAcqusition.loadDisparityMapFromTIF(tifFileName+".tif")
        testResult = np.sum(test_disparityMap - disparityMap)
        print(testResult)
    return
    
if __name__ == "__main__":
    folderPath = "/home/xwk/projects/trackdlo/data/acquiredData/20230516_Configurations_labeled/20230516_115857_arena/"
    convert_npy_to_tif(folderPath)
