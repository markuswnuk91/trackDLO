import os, sys
import datetime
import cv2
try:
    sys.path.append(os.getcwd().replace("/app", ""))
    from src.sensing.dataAcquisition import DataAcquisition
    from src.robot.franka import FrankaEmikaPanda
except:
    print("Imports for application acquireStereoDataSetsAndRobotPose failed.")
    raise

# save path: if None default path is used
savePath = None
defaultPath = "data/acquired_data/"
fps = 30 # maximum fps the application will display images
method = 'auto' # "manual": acqusition on pressing key; "auto": continous acquisiton (video)

def generateFolderPath(path):
    now = datetime.datetime.now()
    date_string = now.strftime("%Y%m%d")
    time_string = now.strftime("%H%M%S")
    newFolderName = date_string + '_' + time_string + '_' + "DataSet"
    newFolderPath = path + newFolderName + "/"
    return newFolderPath

if __name__ == "__main__":
    if savePath is None:
        folderPath = generateFolderPath(defaultPath)
        isExist = os.path.exists(folderPath)
        if not isExist:
            os.makedirs(folderPath)

    # connect to robot 
    robot = FrankaEmikaPanda()
    #connect to camera
    dataAcquistion = DataAcquisition(folderPath)
    transfer = dataAcquistion.setupAsyncConnection()
    dataSetCounter = 0
    run = True
    waitTime = int(1000/fps) # waitTime [ms] = 1000ms / fps
    while run:
        try:
            image_set = dataAcquistion.acquireImageSet(transfer)
            rgb_image = dataAcquistion.getRGBDataFromImageSet(image_set)
            cv2.imshow("RGB image", cv2.resize(rgb_image, None, fx=.25, fy=.25))
            key = cv2.waitKey(waitTime)
            if key == 27:#if ESC is pressed, exit loop
                cv2.destroyAllWindows()
                print("stopped data acquisition.")
                break
            elif (method == "auto") or (key != -1):
                if dataSetCounter==0:
                    dataAcquistion.saveCameraParameters(folderPath)
                stereoDataSet = dataAcquistion.getStereoDataFromImageSet(image_set)
                robotState = robot.getRobotState()
                #generate unique identifier for dataset
                date_time_string = dataAcquistion.generateIdentifier()
                fileNameRGB = date_time_string + "_image_rgb"
                fileNameDisparityMap = date_time_string + "_map_disparity"
                fileNameDisparityImage = date_time_string + "_image_disparity"
                fileNameRobotState= date_time_string + "_robot_state"
                dataAcquistion.saveStereoData(rgb_image = stereoDataSet[0],disparityMap = stereoDataSet[1], folderPath = folderPath, filename_rgbImage = fileNameRGB, filename_disparityMap = fileNameDisparityMap, filename_disparityImage = fileNameDisparityImage)
                dataAcquistion.saveRobotState(robotState, folderPath = folderPath, fileName=fileNameRobotState)
                dataSetCounter += 1
                print("Acquired data sets: {}".format(dataSetCounter)) 
        except:
            cv2.destroyAllWindows()
            print("Stopped data acquisition due to exception")
            run = False
            break
