import os, sys
import datetime
import cv2
import shutil
try:
    sys.path.append(os.getcwd().replace("/app", ""))
    from src.sensing.dataAcquisition import DataAcquisition
    from src.robot.franka import FrankaEmikaPanda
except:
    print("Imports for application acquireStereoDataSetsAndRobotPose failed.")
    raise

# TO BE CHANGED BEFORE ACQUISITION
model = "singleDLO" # "singleDLO, partial, modelY, arena"
fps = 30 # maximum fps the application will display images
method = 'manual' # "manual": acqusition on pressing key; "auto": continous acquisiton (video)
saveRobotPose = False # if robot pose should also be saved
saveSetupParameters = False # setup parameters (fixture positions, grasping points) should be saved
saveRootDirectory = "data/acquiredData/Configurations_mounted/"
dataSetFolderName = model
# not to be configured
modelParameterPath = "src/evaluation/bdloDesciptions/" + model + "/"
graspingPositionsParameterPath = "src/evaluation/setupDesciptions/roboticManipulationScenario/" + model + "/"
setupParameterPath = "src/evaluation/setupDesciptions/manualManipulationScenario/"

calibrationParameterFilePath = "config/calibration/calibrationParameters.json"
now = datetime.datetime.now()
date_string = now.strftime("%Y%m%d")
time_string = now.strftime("%H%M%S")

def generateFolderPath(path, folderName = None):
    now = datetime.datetime.now()
    date_string = now.strftime("%Y%m%d")
    time_string = now.strftime("%H%M%S")
    if folderName is None:
        newFolderName = date_string + '_' + time_string + '_' + "DataSet"
    else:
        newFolderName = date_string + '_' + time_string + '_' + folderName
    newFolderPath = path + newFolderName + "/"
    return newFolderPath

if __name__ == "__main__":
    if saveRootDirectory is None:
        saveRootDirectory = "data/acquiredData/"

    if dataSetFolderName is None:
        folderPath = generateFolderPath(saveRootDirectory)
    else:
        folderPath = generateFolderPath(saveRootDirectory, dataSetFolderName)

    isExist = os.path.exists(folderPath)
    if not isExist:
        os.makedirs(folderPath)
    folderPath_imagedata = folderPath + "data/"
    folderPath_imagedata_exists = os.path.exists(folderPath_imagedata)
    if not folderPath_imagedata_exists:
            os.makedirs(folderPath_imagedata)

    if saveRobotPose:
        # connect to robot 
        robot = FrankaEmikaPanda()
    #connect to camera
    dataAcquistion = DataAcquisition(folderPath)
    transfer = dataAcquistion.setupAsyncConnection()
    dataSetCounter = 0
    run = True
    waitTime = int(1000/fps) # waitTime [ms] = 1000ms / fps

    acquiring = False
    if method == "auto":
        print("Hit ENTER to start acquisition ...")
    while run:
        try:
            image_set = dataAcquistion.acquireImageSet(transfer)
            rgb_image = dataAcquistion.getRGBDataFromImageSet(image_set)
            cv2.imshow("RGB image", cv2.resize(rgb_image, None, fx=0.8, fy=0.8))
            key = cv2.waitKey(waitTime)
            if key == 27:#if ESC is pressed, exit loop
                cv2.destroyAllWindows()
                print("stopped data acquisition.")
                break
            elif (method == "manual") and (key != -1):
                acquiring = True
            elif (method == "auto") and (key == 13):
                acquiring = True
            if acquiring:    
                if dataSetCounter==0:
                    #add meta data for data set
                    dataAcquistion.saveCameraParameters(folderPath)
                    shutil.copy(calibrationParameterFilePath, folderPath)
                    shutil.copy(modelParameterPath + "model.json", folderPath)
                    shutil.copy(modelParameterPath + "model.png", folderPath)
                    shutil.copy(modelParameterPath + "labels.png", folderPath)
                    if saveSetupParameters:
                        shutil.copy(setupParameterPath + "setup.json", folderPath)
                        shutil.copy(setupParameterPath + "setup.png", folderPath)
                        shutil.copy(graspingPositionsParameterPath + "graspingPositions.json", folderPath)
                        shutil.copy(graspingPositionsParameterPath + "graspingPositions.png", folderPath)
                stereoDataSet = dataAcquistion.getStereoDataFromImageSet(image_set)
                if saveRobotPose:
                    robotState = robot.getRobotState()
                #generate unique identifier for dataset
                date_time_string = dataAcquistion.generateIdentifier()
                fileNameRGB = date_time_string + "_image_rgb"
                fileNameDisparityMap = date_time_string + "_map_disparity"
                fileNameDisparityImage = date_time_string + "_image_disparity"
                fileNameRobotState= date_time_string + "_robot_state"
                dataAcquistion.saveStereoData(rgb_image = stereoDataSet[0],disparityMap = stereoDataSet[1], disparity_image = stereoDataSet[2], folderPath = folderPath_imagedata, filename_rgbImage = fileNameRGB, filename_disparityMap = fileNameDisparityMap, filename_disparityImage = fileNameDisparityImage)
                if saveRobotPose:
                    dataAcquistion.saveRobotState(robotState, folderPath = folderPath_imagedata, fileName=fileNameRobotState)
                dataSetCounter += 1
                print("Acquired data sets: {}".format(dataSetCounter)) 
                if method == "manual":
                    acquiring = False
        except:
            cv2.destroyAllWindows()
            print("Stopped data acquisition due to exception")
            run = False
            break

