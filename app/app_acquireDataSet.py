import os, sys
import datetime

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

def acquireData():
    if savePath is None:
        folderPath = generateFolderPath(defaultPath)
        isExist = os.path.exists(folderPath)
        if not isExist:
            os.makedirs(folderPath)
    else:
        folderPath = savePath
    dataAcquistion = DataAcquisition(folderPath)
    dataAcquistion.recordStereoDataSet()

if __name__ == "__main__":
    acquireData()