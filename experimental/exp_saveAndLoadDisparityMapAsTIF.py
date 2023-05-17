import os, sys
import datetime
import numpy as np
try:
    sys.path.append(os.getcwd().replace("/tests", ""))
    from src.sensing.cameraInterface import CameraInterface
    from src.sensing.dataAcquisition import DataAcquisition
except:
    print("Imports for testing camera interface failed.")
    raise

# save path: if None default path is used
savePath = None
defaultPath = "data/acquiredData/"


def generateFolderPath(path):
    now = datetime.datetime.now()
    date_string = now.strftime("%Y%m%d")
    time_string = now.strftime("%H%M%S")
    newFolderName = date_string + '_' + time_string + '_' + "DataSet"
    newFolderPath = path + newFolderName + "/"
    return newFolderPath
    
if __name__ == "__main__":
    # acquire data
    if savePath is None:
        folderPath = generateFolderPath(defaultPath)
    isExist = os.path.exists(folderPath)
    if not isExist:
        os.makedirs(folderPath)
    folderPath_imagedata = folderPath + "data/"
    folderPath_imagedata_exists = os.path.exists(folderPath_imagedata)
    if not folderPath_imagedata_exists:
            os.makedirs(folderPath_imagedata)
    else:
        folderPath = savePath
    dataAcquistion = DataAcquisition(folderPath)
    dataAcquistion.recordStereoDataSet(saveDisparityMap=True)

    # load data
    dataAcquistion.defaultLoadFolderPath = folderPath
    dataAcquistion.defaultLoadFolderPath_Data = folderPath +"data/"
    fileName_NPY = dataAcquistion.getDataSetFileNames_NPY()[0]
    disparityMap_fromBinary = dataAcquistion.loadNumpyArrayFromBinary(fileName_NPY)

    fileName_TIF = dataAcquistion.getDataSetFileNames_TIF()[0]
    disparityMap_fromTif = dataAcquistion.loadDisparityMapFromTIF(fileName_TIF)

    comparisonResult = np.sum(disparityMap_fromBinary - disparityMap_fromTif)
    print(comparisonResult)
