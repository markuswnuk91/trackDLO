import numpy as np


class cost:

    def __init__(self, Checkerboard_T_Cam, Base_T_EE):
        self.Checkerboard_T_Cam = Checkerboard_T_Cam
        self.Base_T_EE = Base_T_EE
        self.numberOfImages = len(self.Base_T_EE)

    def costFun(self, q):

        EE_T_Checkerboard = self.getEE_T_Checkerboard(q)

        # Setup Transform from Kamera to Robot

        Cam_T_Robot = self.getCam_T_Robot(q)

        cost = 0

        for i in range(len(self.Base_T_EE)):
            error = self.Base_T_EE[i] @ EE_T_Checkerboard @ self.Checkerboard_T_Cam[i] @ Cam_T_Robot - \
                np.identity((4))

            cost += np.sum(np.square((error)))

        return cost

    def costFunSingle(self, q, i):
        EE_T_Checkerboard = self.getEE_T_Checkerboard(q)

        # Setup Transform from Kamera to Robot

        Cam_T_Robot = self.getCam_T_Robot(q)
        return self.Base_T_EE[i] @ EE_T_Checkerboard @ self.Checkerboard_T_Cam[i] @ Cam_T_Robot

    def getEE_T_Checkerboard(self, q):
        EE_T_Checkerboard = np.vstack((
            [
                np.hstack((self.getRz(q[2]) @ self.getRy(q[1]) @ self.getRx(q[0]),
                           np.array([[q[3]], [q[4]], [q[5]]]))),
                [0, 0, 0, 1],
            ]
        ))
        return EE_T_Checkerboard

    def getCam_T_Robot(self, q):
        Cam_T_Robot = np.vstack((
            [
                np.hstack((self.getRz(q[8]) @ self.getRy(q[7]) @ self.getRx(q[6]),
                           np.array([[q[9]], [q[10]], [q[11]]]))),
                [0, 0, 0, 1],
            ]
        ))
        return Cam_T_Robot

    def getRx(self, alpha):
        # Helper Functions
        Rx = np.array(([[1, 0, 0], [0, np.cos(alpha), -np.sin(alpha)], [
            0, np.sin(alpha), np.cos(alpha)]]))
        return Rx

    def getRy(self, alpha):
        # Helper Functions
        Ry = np.array(([[np.cos(alpha), 0, np.sin(alpha)],
                        [0, 1, 0], [-np.sin(alpha), 0, np.cos(alpha)]]))
        return Ry

    def getRz(self, alpha):
        # Helper Functions
        Rz = np.array(([[np.cos(alpha), -np.sin(alpha), 0],
                        [np.sin(alpha), np.cos(alpha), 0], [0, 0, 1]]))
        return Rz
