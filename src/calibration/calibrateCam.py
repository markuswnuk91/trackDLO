import json
from scipy import optimize
from scipy.spatial.transform import Rotation as R
import numpy as np
import cv2
from cv2 import aruco
import visiontransfer
import os
import sys
import argparse

sys.path.append(os.getcwd().replace("/src/Calibration", ""))

try:
    from src.Calibration.costFunCalib import cost
    # from src.Robot.readOutRobot import myRobot
    from src.robot.franka import FrankaEmikaPanda
except:
    print("Import error")
    raise


class calibrateCamera:
    translations = []
    rotations = []

    M1_nerian = [
        [2.3300757371615437e03, 0.0, 1.2258353990366115e03],
        [0.0, 2.3273181724097385e03, 1.0386921034960960e03],
        [0.0, 0.0, 1.0],
    ]
    D1_nerian = [
        -6.3416442150127303e-02,
        1.2502716905238478e-01,
        8.6872791605854284e-04,
        -8.1099397816424186e-04,
        -5.3872915654436374e-02,
    ]

    def __init__(self):
        """
        Class to make the Camera Calibration
        """

    def getTVecsAndRVecs(self, imgs):

        objpoints = []
        imgpoints = []

        cornerlist = []
        idlist = []
        skipImageIndex = []

        # Scale the 3D object points
        # Scale tvecs by 1/1000 to have m instead of mm 24.35 measured
        scaleFactorChessBoardSquare = 24.35 * 1 / 1000

        nx = 9
        ny = 6

        charuco = False

        for img in imgs:

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

            if ret:
                objp = np.zeros((nx * ny, 3), np.float32)
                objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

                objp = objp * scaleFactorChessBoardSquare

                # termination criteria
                criteria = (
                    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    30,
                    0.001,
                )
                corners2 = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1), criteria)

                # Find the rotation and translation vectors.
                ret, rvecs_3, tvecs_3, inliers_3 = cv2.solvePnPRansac(
                    objp,
                    corners2,
                    np.asarray(self.M1_nerian),
                    np.asarray(self.D1_nerian),
                )

                imgpoints.append(corners)
                objpoints.append(objp)

                # For testing
                self.translations.append(tvecs_3)
                self.rotations.append(rvecs_3)

            else:
                # Check if ARUCO Marker
                # Try Charuco
                aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_1000)

                # ChAruco board variables
                CHARUCOBOARD_ROWCOUNT = 7
                CHARUCOBOARD_COLCOUNT = 5

                # Create constants to be passed into OpenCV and Aruco methods
                board = aruco.CharucoBoard_create(
                    squaresX=CHARUCOBOARD_COLCOUNT,
                    squaresY=CHARUCOBOARD_ROWCOUNT,
                    squareLength=0.04,
                    markerLength=0.02,
                    dictionary=aruco_dict,
                )

                corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(
                    gray, aruco_dict
                )

                if len(corners) > 0:
                    charuco = True
                    res2 = cv2.aruco.interpolateCornersCharuco(
                        corners, ids, gray, board
                    )
                    decimator = 0

                    if (
                        res2[1] is not None
                        and res2[2] is not None
                        and len(res2[1]) > 3
                        and decimator % 1 == 0
                    ):
                        if len(res2[1]) > 6:

                            # res2[1] == corners
                            # res2[2] == ids
                            cornerlist.append(res2[1])
                            idlist.append(res2[2])
                        else:
                            skipImageIndex.append(
                                len(cornerlist) + len(skipImageIndex))
                            print(
                                "Consider deleting Image {}, corners < 6, will not be used for calculation.".format(
                                    len(cornerlist)
                                )
                            )
                    else:
                        skipImageIndex.append(
                            len(cornerlist) + len(skipImageIndex))
                        print(
                            "Consider deleting Image {}, highly occluded image, will not be used for calculation.".format(
                                len(cornerlist)
                            )
                        )
                else:
                    print("No Corners Found!")

        if charuco:
            (
                ret,
                mtx,
                dist,
                rvecs,
                tvecs,
                stdDeviationsIntrinsics,
                stdDeviationsExtrinsics,
                perViewErrors,
            ) = aruco.calibrateCameraCharucoExtended(
                charucoCorners=cornerlist,
                charucoIds=idlist,
                board=board,
                imageSize=gray.shape[::-1],
                cameraMatrix=np.asarray(self.M1_nerian),
                distCoeffs=np.asarray(self.D1_nerian),
            )

            self.M1 = mtx
            self.D1 = dist

            self.translations = tvecs
            self.rotations = rvecs

        if len(objpoints) > 5 and not charuco:

            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                objpoints,
                imgpoints,
                gray.shape[::-1],
                np.asarray(self.M1_nerian),
                np.asarray(self.D1_nerian),
            )

            self.M1 = mtx
            self.D1 = dist

            self.translations = tvecs
            self.rotations = rvecs

        return skipImageIndex

    def draw(self, img, corners, imgpts):
        # Draw a Cube
        imgpts = np.int32(imgpts).reshape(-1, 2)

        # draw ground floor in green
        img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -3)

        # draw pillars in blue color
        for i, j in zip(range(4), range(4, 8)):
            img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255), 3)

        # draw top layer in red color
        img = cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)

        return img

    def makePictureAndCheckCheckerboard(self):

        maxTries = 100
        itr = 0

        subImgs = self.communication.createSub(
            self.communication.cameraImagesAddress)

        noImageReceived = True
        while noImageReceived:
            try:
                imgdata = subImgs.recv_msgpack()[0]
                noImageReceived = False
            except:
                itr += 1
                if itr == maxTries:
                    print("Failed to receive Images for saving.")
                    return False, None
                pass

        if len(imgdata.shape) == 2:
            # Gray Color
            img = cv2.cvtColor(imgdata, cv2.COLOR_GRAY2BGR)

        if len(imgdata.shape) == 3:
            # Color Image
            img = cv2.cvtColor(imgdata, cv2.COLOR_RGB2BGR)

        # cv2.imshow("ISW ARENA CALIBRATION TOOL",img)
        # cv2.waitKey()

        print("Searching for Corners")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        nx = 9
        ny = 6

        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        if ret:
            print("Corners found!")

            # Save Image
            itr = 0
            if itr == 0:
                try:
                    print("Try to create Recording directory")
                    os.mkdir("./CalibRecordings")
                    print("Created directory ", os.getcwd(), "/CalibRecordings")
                except OSError:
                    path = os.getcwd()
                    print("Directory ", path, "/CalibRecordings already exists")

            name = "img_" + str(itr) + ".png"
            name_with_corners = "img_corner_" + str(itr) + ".png"
            while os.path.exists("./CalibRecordings/" + name):
                itr += 1
                name = "img_" + str(itr) + ".png"
                name_with_corners = "img_corner_" + str(itr) + ".png"

            print("Saving... ", str(name))
            cv2.imwrite("./CalibRecordings/" + name, gray)
            cv2.drawChessboardCorners(img, (nx, ny), corners, ret)

            # Draw also a coordinate system to check the rvecs and tvecs
            # termination criteria
            criteria = (
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                30,
                0.001,
            )
            corners2 = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria)
            objp = np.zeros((nx * ny, 3), np.float32)
            objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

            objp = objp * 24.35 / 1000

            # Find the rotation and translation vectors.
            ret, rvecs, tvecs, inliers = cv2.solvePnPRansac(
                objp,
                corners2,
                np.asarray(self.M1_nerian_color),
                np.asarray(self.D1_nerian_color),
            )

            sideCube = 0.05
            axis = np.float32(
                [
                    [0, 0, 0],
                    [0, sideCube, 0],
                    [sideCube, sideCube, 0],
                    [sideCube, 0, 0],
                    [0, 0, -sideCube],
                    [0, sideCube, -sideCube],
                    [sideCube, sideCube, -sideCube],
                    [sideCube, 0, -sideCube],
                ]
            )

            # project 3D points to image plane
            imgpts, jac = cv2.projectPoints(
                axis,
                rvecs,
                tvecs,
                np.asarray(self.M1_nerian_color),
                np.asarray(self.D1_nerian_color),
            )

            img = self.draw(img, corners2, imgpts)
            print("Saving... ", str(name_with_corners))

            cv2.imwrite("./CalibRecordings/" + name_with_corners, img)
            return True, img
        else:
            print("Coudln't find Chessboard, invalid picture")
            return False, None

    def makePicture(self, offline=False, img=None):
        """
        Make a picture, return true if it worked
        false if the chessboard isn't visible

        If given a img skip the camera
        """
        noDevice = True
        if not offline:
            # if self.streamViewerActive:
            #     # Toggle Stream Viewer off
            #     self.streamViewer()
            device_enum = visiontransfer.DeviceEnumeration()
            devices = device_enum.discover_devices()
            while noDevice:

                if len(devices) < 1:
                    print("No devices founds")
                    # return False
                    devices = device_enum.discover_devices()
                if len(devices) >= 1:
                    noDevice = False

            print("Found device")
            selected_device = 0

            device = devices[selected_device]

            transfer = visiontransfer.AsyncTransfer(device)

            image_set = transfer.collect_received_image_set()
            imgdata = image_set.get_pixel_data(0, force8bit=True)

            if len(imgdata.shape) == 2:
                # Gray Color
                img = cv2.cvtColor(imgdata, cv2.COLOR_GRAY2BGR)

            if len(imgdata.shape) == 3:
                # Color Image
                img = cv2.cvtColor(imgdata, cv2.COLOR_RGB2BGR)

            # cv2.imshow("ISW ARENA CALIBRATION TOOL",img)
            # cv2.waitKey()

            print("Searching for Corners")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        nx = 9
        ny = 6

        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        if ret:
            print("Corners found!")

            # Save Image
            if not offline:
                itr = 0
                if itr == 0:
                    try:
                        print("Try to create Recording directory")
                        os.mkdir("./CalibRecordings")
                        print("Created directory ",
                              os.getcwd(), "/CalibRecordings")
                    except OSError:
                        path = os.getcwd()
                        print("Directory ", path,
                              "/CalibRecordings already exists")

                name = "img_" + str(itr) + ".png"
                name_with_corners = "img_corner_" + str(itr) + ".png"
                while os.path.exists("./CalibRecordings/" + name):
                    itr += 1
                    name = "img_" + str(itr) + ".png"
                    name_with_corners = "img_corner_" + str(itr) + ".png"

                print("Saving... ", str(name))
                cv2.imwrite("./CalibRecordings/" + name, gray)
                cv2.drawChessboardCorners(img, (nx, ny), corners, ret)

                # Draw also a coordinate system to check the rvecs and tvecs

                # termination criteria
                criteria = (
                    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    30,
                    0.001,
                )
                corners2 = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1), criteria)
                objp = np.zeros((nx * ny, 3), np.float32)
                objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

                objp = objp * 24.35 / 1000

                # Find the rotation and translation vectors.
                ret, rvecs, tvecs, inliers = cv2.solvePnPRansac(
                    objp,
                    corners2,
                    np.asarray(self.M1_nerian),
                    np.asarray(self.D1_nerian),
                )

                sideCube = 0.1
                axis = np.float32(
                    [
                        [0, 0, 0],
                        [0, sideCube, 0],
                        [sideCube, sideCube, 0],
                        [sideCube, 0, 0],
                        [0, 0, -sideCube],
                        [0, sideCube, -sideCube],
                        [sideCube, sideCube, -sideCube],
                        [sideCube, 0, -sideCube],
                    ]
                )

                # project 3D points to image plane
                imgpts, jac = cv2.projectPoints(
                    axis,
                    rvecs,
                    tvecs,
                    np.asarray(self.M1_nerian),
                    np.asarray(self.D1_nerian),
                )

                img = self.draw(img, corners2, imgpts)
                print("Saving... ", str(name_with_corners))

                cv2.imwrite("./CalibRecordings/" + name_with_corners, img)

            return True, img

        else:
            print("Try finding a charuco board")
            # Try Charuco
            aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_1000)

            # ChAruco board variables
            CHARUCOBOARD_ROWCOUNT = 7
            CHARUCOBOARD_COLCOUNT = 5

            # Create constants to be passed into OpenCV and Aruco methods
            board = aruco.CharucoBoard_create(
                squaresX=CHARUCOBOARD_COLCOUNT,
                squaresY=CHARUCOBOARD_ROWCOUNT,
                squareLength=0.04,
                markerLength=0.02,
                dictionary=aruco_dict,
            )

            corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(
                gray, aruco_dict)

            if len(corners) > 0:
                print("Found Charuco Board")

                # Save Image
                itr = 0

                if self.saveLocal and not offline:
                    if itr == 0:
                        try:
                            print("Try to create Recording directory")
                            os.mkdir("./CalibRecordings")
                            print("Created directory ",
                                  os.getcwd(), "/CalibRecordings")
                        except OSError:
                            path = os.getcwd()
                            print("Directory ", path,
                                  "/CalibRecordings already exists")

                name = "img_" + str(itr) + ".png"
                name_with_corners = "img_corner_" + str(itr) + ".png"
                while os.path.exists("./CalibRecordings/" + name):
                    itr += 1
                    name = "img_" + str(itr) + ".png"
                    name_with_corners = "img_corner_" + str(itr) + ".png"

                if self.saveLocal:
                    print("Saving... ", str(name))
                    cv2.imwrite("./CalibRecordings/" + name, gray)
                    aruco.drawDetectedCornersCharuco(img, np.asarray(corners))

                # SUB PIXEL DETECTION
                # for corner in corners:
                #     cv2.cornerSubPix(
                #         gray,
                #         corner,
                #         winSize=(3, 3),
                #         zeroZone=(-1, -1),
                #         criteria=criteria,
                #     )

                # img = self.draw(img, corners2, imgpts)
                if self.saveLocal:

                    print("Saving... ", str(name_with_corners))
                    cv2.imwrite("./CalibRecordings/" + name_with_corners, img)

                res2 = cv2.aruco.interpolateCornersCharuco(
                    corners, ids, gray, board)
                decimator = 0
                if (
                    res2[1] is not None
                    and res2[2] is not None
                    and len(res2[1]) > 3
                    and decimator % 1 == 0
                ):
                    # res2[1] == corners
                    # res2[2] == ids

                    pass
                return True, img

            else:

                print("Coudln't find Chessboard nor Charuco board, invalid picture")
                return False, None

    def get4x4(self, trans, rotVec):
        """
        Return 4x4 with Rodrigues Vec
        """
        rot = R.from_rotvec(np.reshape(rotVec, (3,)))
        T = np.identity((4))
        T[:3, :3] = rot.as_matrix()
        T[:3, 3] = np.reshape(trans, (1, 3))
        return T

    def getInv4x4From4x4(self, mat2):
        """
        Return inv 4x4

        [R_T -R_T * t]
        [0          1]
        """
        mat = np.copy(mat2)
        R_T = mat[:3, :3].T
        mat[:3, :3] = R_T
        mat[:3, 3] = -R_T @ mat[:3, 3]
        return mat

    def getInv4x4(self, trans, rotVec):
        """
        Return inv 4x4 with Rodrigues Vec

        [R_T -R_T * t]
        [0          1]
        """
        rot = R.from_rotvec(np.reshape(rotVec, (3,)))
        R_T = rot.as_matrix().T
        T = np.identity((4))
        T[:3, :3] = R_T
        trans = -R_T @ trans
        T[:3, 3] = np.reshape(trans, (1, 3))
        return T

    def removeLastImage(self):
        """
        Removes the values saved from the last image
        """
        self.translations.pop(len(self.translations) - 1)
        self.rotations.pop(len(self.rotations) - 1)

    def calculateCameraCalibrationScipy(self, Base_T_EE, saveToJson=True):
        """
        Seems to work
        """
        print("Solving with scipy ...")
        # ========================================================================
        # Setup Rotation Problem
        Checkerboard_T_Cam = []

        for i in range(len(Base_T_EE)):
            Checkerboard_T_Cam.append(
                self.getInv4x4(self.translations[i], self.rotations[i])
            )

        mCost = cost(Checkerboard_T_Cam, Base_T_EE)
        minimum = optimize.least_squares(
            mCost.costFun,
            np.zeros((12,)),
            jac="3-point",
            method="trf",
            xtol=1e-12,
            ftol=1e-12,
        )

        curr_x = minimum.x

        filterOutliers = True
        if filterOutliers:
            print("Filter outliers is true, start filtering...")
            cost_list = []
            for i in range(len(Base_T_EE)):
                cost4x4_i = mCost.costFunSingle(minimum.x, i)
                cost_list.append(np.sum(np.square(cost4x4_i - np.identity(4))))

            cost_median = sum(cost_list) / len(cost_list)
            itr = 0
            for singleCost in cost_list:
                if singleCost > cost_median:
                    index = cost_list.index(singleCost)
                    del mCost.Checkerboard_T_Cam[index - itr]
                    del mCost.Base_T_EE[index - itr]
                    itr += 1
            print("Deleted ", itr, " Outliers, continue to calculate minimum ...")

            minimum = optimize.least_squares(
                mCost.costFun,
                curr_x,
                jac="3-point",
                method="trf",
                xtol=1e-12,
                ftol=1e-12,
            )

        print("Found minimum at", minimum.x)

        if saveToJson:
            # Implement that it saves the calibration to a JSON File
            pass

        return mCost.getCam_T_Robot(minimum.x), mCost.getEE_T_Checkerboard(minimum.x)

    def readFromDirectory(self, path):
        """
        Read directory and search path for
        pos_x.json
        img_x.json
        """
        itr = 0
        name = "pos_" + str(itr) + ".json"
        name_img = "img_" + str(itr) + ".png"
        Base_T_EE = []
        img_total = []

        while os.path.exists(path + name):
            with open(path + name) as f:
                try:
                    # Base_T_EE.append(np.reshape(json.load(f)["Base_T_EE"], (4, 4)).T)
                    # -- Arena Custom --:
                    tmp = np.reshape(json.load(f)["Base_T_EE"], (4, 4))
                    # tmp[:3, 3] = tmp[:3, 3] * 1 / 1000
                    Base_T_EE.append(tmp)

                    img = cv2.imread(path + name_img)
                    img_total.append(img)
                    success, img = self.makePicture(offline=True, img=img)
                    if not success:
                        print(
                            "Reading images failed. Delete image: ",
                            path + name_img,
                            " and try again.",
                        )
                        sys.exit(1)
                except:
                    print(
                        "Invalid input, skipping: ", str(
                            name), " and ", str(name_img)
                    )

                itr += 1
                name = "pos_" + str(itr) + ".json"
                name_img = "img_" + str(itr) + ".png"

        skipImageIndex = self.getTVecsAndRVecs(img_total)

        mask = np.ones(len(Base_T_EE), np.bool)
        mask[skipImageIndex] = 0
        filtered_Base_T_EE = np.asarray(Base_T_EE)[mask].tolist()

        Base_T_EE = filtered_Base_T_EE

        print("Reading in complete.")
        return Base_T_EE

    def saveEEPosition(self, EE_pos):
        """
        Save EE Position given as a json
        """
        itr = 0

        name = "pos_" + str(itr) + ".json"
        while os.path.exists("./CalibRecordings/" + name):
            itr += 1
            name = "pos_" + str(itr) + ".json"
        print("Saving ...", str(name))
        with open("./CalibRecordings/" + name, "w") as outfile:
            json.dump(EE_pos, outfile)

    def saveJson(self, data):
        """
        Save the Json File
        """
        with open("src/Calibration/Calibration.json", "w") as output:
            output.write(json.dumps(data, indent=4))


def getEEPosition(robotInterface, saveLocal=True):

    mydata = {}
    mydata["Base_T_EE"] = robotInterface.getO_T_EE().tolist()

    if saveLocal:
        itr = 0

        name = "pos_" + str(itr) + ".json"
        while os.path.exists("./CalibRecordings/" + name):
            itr += 1
            name = "pos_" + str(itr) + ".json"

        with open("./CalibRecordings/" + name, "w") as outfile:
            json.dump(mydata, outfile, indent=4)

    return mydata["Base_T_EE"]


def saveCalib(M1, D1, Cam2Robot, Robo2Cam, EE2Checkerboard):

    mydata = {}
    mydata["M1"] = M1.tolist()
    mydata["D1"] = D1.tolist()
    mydata["Cam2Robot"] = Cam2Robot.tolist()
    mydata["Robo2Cam"] = Robo2Cam.tolist()
    mydata["EE2Checkerboard"] = EE2Checkerboard.tolist()

    with open("./data/" + "calibration.json", "w") as outfile:
        json.dump(mydata, outfile, indent=4)


def main():
    """
    Test Function
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "online_offline", help="read from directory -> offline, prepare with robot -> online", type=str)
    args = parser.parse_args()

    if args.online_offline == "offline":
        print("*********\nPerforming offline Test\n*********\n")

        test = calibrateCamera()
        Base_T_EE = test.readFromDirectory("./CalibRecordings/")

        print("Total number of Poses: ", str(len(Base_T_EE)))
        print("Start calibration")

        Cam_T_Robot, EE_T_Checkerboard = test.calculateCameraCalibrationScipy(
            Base_T_EE)

        print("\nM1:\n", np.array2string(test.M1, separator=","))
        print("\nD1:\n", np.array2string(test.D1, separator=","))

        print("\nSolution Cam_T_Robot:\n",
              np.array2string(Cam_T_Robot, separator=","))
        print(
            "Sol robo_2_cam: \n",
            np.array2string(np.linalg.inv(Cam_T_Robot), separator=","),
        )
        print(
            "\nSolution EE_T_Checkerboard:\n",
            np.array2string(EE_T_Checkerboard, separator=","),
        )

        myDict = {}
        myDict["Cam_T_Robot"] = Cam_T_Robot.tolist()
        myDict["EE_T_Checkerboard"] = EE_T_Checkerboard.tolist()
        saveCalib(test.M1, test.D1, Cam_T_Robot,
                  np.linalg.inv(Cam_T_Robot), EE_T_Checkerboard)

    if args.online_offline == "online":

        print("*********\nPerforming online Test\n*********\n")
        camCal = calibrateCamera()

        Base_T_EE = []
        imgs = []
        skipToNext = False

        robotInterface = FrankaEmikaPanda()
        while True:
            a = input("\nHit Enter to capture image, write stop to stop: ")
            if a == "stop":
                break
            print("Making picture ...")
            foundChessboard, img = camCal.makePicture()
            if not foundChessboard:
                print("Invalid picture, Chessboard not found. Triggering again ...")
                reFoundChessboard, img = camCal.makePicture()

                if reFoundChessboard:
                    print("Chessboard found!")
                else:
                    skipToNext = True

            if not skipToNext:
                print("Chessboard found!")
                imgs.append(img)
                print("Get Robot EE Position ...")
                try:
                    myPos4x4 = getEEPosition(robotInterface)
                    print("Found EE Pos, saving: ", myPos4x4)
                    Base_T_EE.append(myPos4x4)

                except ValueError:
                    print("I DONT KNOW WHATS WRONG")
                    camCal.removeLastImage()

            else:
                print("Try a new position ...")
                skipToNext = False


if __name__ == "__main__":
    main()
