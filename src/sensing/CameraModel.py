import sys, os
import numpy as np
import numbers
import random
from warnings import warn

try:
    sys.path.append(os.getcwd().replace("/src/sensing", ""))
except:
    print("Imports for CameraModel failed.")
    raise


class CameraModel(object):
    """Camera model, to create point cloud data from a DLO Model.
    The model is based on geometric approximations to determine the surface points of the DLO visible to the camera.
    For each observed point we assume sensor noise, based on the noise model from the paper
    Nguyen et al., Modeling Kinect Sensor Noise for Improved 3D Reconstruction and Tracking, 2012 Second Joint 3DIM/3DPVT Conference: 3D Imaging, Modeling, Processing, Visualization & Transmission, 2012

    Attributes
        ----------
        X: Nx3 np.array
            positions of the DLO representing its skeleton line expressed in some fixed reference coordinate system

        localTangents:
            Nx3 np.array
            unit vector representing the local tangent of the DLO's skeleton line.

        radius: float
            radius of the DLO.

        camTransform: 3x1 np.array
            Position of the camera in the reference coordinate system the DLO's positions are expressed
    """

    def __init__(
        self,
        camTransform=None,
        X=None,
        localTangents=None,
        radius=None,
        laterNoiseGradient=None,
        axialNoiseOffset=None,
        axialNoiseGradient=None,
        axialNoiseShiftFactor=None,
        *args,
        **kwargs
    ):

        if type(camTransform) is not np.ndarray or camTransform.ndim != 2:
            raise ValueError("The cam transform must be a 2D numpy array.")
        elif camTransform.shape != (4, 4):
            raise ValueError("The cam transform must be a 4x4 homogenous matrix.")
        elif (
            np.sum(np.linalg.inv(camTransform) @ camTransform - np.eye(4)) >= 10e-5
        ) or (np.linalg.det(camTransform) - 1 >= 10e-5):
            raise ValueError(
                "The cam transform must be a homogenous matrix. Obtained a error of {} between forward and inverse and a determinant of {}.".format(
                    np.sum(np.linalg.inv(camTransform) @ camTransform - np.eye(4)),
                    np.linalg.det(camTransform),
                )
            )

        if type(X) is not np.ndarray or X.ndim != 2:
            raise ValueError(
                "The points on the skeleton line (X) must be a 2D numpy array."
            )
        elif X.shape[1] != 3:
            raise ValueError(
                "The points on the skeleton line (X) must be specify xyz values for each point."
            )

        if type(localTangents) is not np.ndarray or localTangents.ndim != 2:
            raise ValueError(
                "The local tangents to the skeleton line must be a 2D numpy array."
            )
        elif localTangents.shape[1] != 3 or localTangents.shape[0] != X.shape[0]:
            raise ValueError(
                "The local tangents to the skeleton line must be specify a 3 dimensional direction vector for each point in X."
            )
        elif np.any(np.linalg.norm(localTangents, axis=1) - 1 > 10e-8):
            warn(
                "Received non-normalized tangent vectors with lengths: {}. Normalizing the tangent vectors.".format(
                    np.linalg.norm(localTangents, axis=1)
                )
            )
            row_sums = localTangents.sum(axis=1)
            localTangents = localTangents / row_sums[:, np.newaxis]

        if radius is not None and (
            not isinstance(radius, numbers.Number) or radius < 0
        ):
            raise ValueError(
                "Expected a positive float for radius instead got: {}".format(radius)
            )
        elif isinstance(radius, numbers.Number) and not isinstance(radius, float):
            warn(
                "Received a non-float value for radius: {}. Casting to float.".format(
                    radius
                )
            )
            radius = float(radius)

        self.camTransform = np.identity(4) if camTransform is None else camTransform
        self.radius = 0.1 if radius is None else radius
        self.camPosition = camTransform[:3, 3]
        self.X = X
        self.localTangents = localTangents

        # values from Nguyen et al., see Figure 6
        self.laterNoiseGradient = (
            0.815 / 585 if laterNoiseGradient is None else laterNoiseGradient
        )
        self.axialNoiseOffset = 0.0012 if axialNoiseOffset is None else axialNoiseOffset
        self.axialNoiseGradient = (
            (0.0019,) if axialNoiseGradient is None else axialNoiseGradient
        )
        self.axialNoiseShiftFactor = (
            (0.4,) if axialNoiseShiftFactor is None else axialNoiseShiftFactor
        )

    def calculateCameraVectors(self, X):
        """Calculates the vector c between a point on the skeleton line and the position of the camera

        Args:
            X (np.array): array of positions on the skeleton line

        Returns:
            C (np.array): array of relative perspective vectors
        """
        C = np.zeros(X.shape)
        for i, x in enumerate(X):
            C[i, :] = self.camPosition - x
        return C

    def calculateCameraNormals(self, X, localTangents):
        """Calculates the local normal n in direction of the camera for each given point of the skeleton line.

        Args:
            X (np.array): array of positions on the skeleton line
            localTangents: array of unit vectors representing the local tangent of the DLO's skeleton line corresponding to the positions in X.
        Returns:
            Nc (np.array): array of relative perspective vectors
        """
        Nc = np.zeros(X.shape)
        C = self.calculateCameraVectors(X)
        for i, x in enumerate(X):
            Nc[i, :] = (
                C[i, :] - np.dot(localTangents[i, :], C[i, :]) * localTangents[i, :]
            )
            Nc[i, :] = Nc[i, :] / np.linalg.norm(Nc[i, :])
            if np.dot(C[i, :], Nc[i, :]) < 0:
                Nc[i, :] = -Nc[i, :]
        return Nc

    def calculateTiltAngle(self, X, localTangents):
        """function to calculate the tilt angle between the relative vector between skeleton line and camera (c) and the local normal in direction of the camera (n)

        Args:
            X (np.array): skeleton line position
            localTangents: array of unit vectors representing the local tangent of the DLO's skeleton line corresponding to the positions in X.
        Returns:
            Theta (np.array): array angles between the skeleton line
        """
        Thetas = np.zeros(X.shape[0])
        C = self.calculateCameraVectors(X)
        Nc = self.calculateCameraNormals(X, localTangents)
        for i, theta in enumerate(Thetas):
            cNormalized = C[i, :] / np.linalg.norm(C[i, :])
            nc = Nc[i, :]
            theta = np.arccos(np.dot(cNormalized, nc))
            Thetas[i] = theta
        return Thetas

    def calculateMaxViewAngle(self, X, localTangents):
        PsiMax = np.pi / 2 * np.ones(X.shape[0])
        Thetas = self.calculateTiltAngle(X, localTangents)
        C = self.calculateCameraVectors(X)
        for i, psi in enumerate(PsiMax):
            PsiMax[i] = np.arccos(self.radius / np.linalg.norm(C[i, :])) * np.cos(
                Thetas[i]
            )
        return PsiMax

    def calculateCameraNoise(self, P):
        """calculates the lateral and axial noise component for points observed by the camera according to the noise model descirbed in the paper
        Nguyen et al., Modeling Kinect Sensor Noise for Improved 3D Reconstruction and Tracking, 2012 Second Joint 3DIM/3DPVT Conference: 3D Imaging, Modeling, Processing, Visualization & Transmission, 2012

        Args:
            P (np.array): Mx3 array of points for which the sensor noise should be computed, werhe M is the number of points.

        Returns:
            sigmaLateral(np.array): Mx1 array of lateral noise values, descibing the lateral uncertainy (x,y component in camera coordinate system)
            sigmaAxial(np.array): Mx1 array of axial noise values, descibing the axial uncertainy (z component in camera coordinate system)
        """
        PInCamCoordinates = (
            np.linalg.inv(self.camTransform)
            @ np.hstack((P, np.ones((P.shape[0], 1)))).T
        ).T[:, :3]
        sigmaLateral = self.laterNoiseGradient * PInCamCoordinates[:, 2]
        sigmaAxial = (
            self.axialNoiseOffset
            + self.axialNoiseGradient
            * (PInCamCoordinates[:, 2] - self.axialNoiseShiftFactor) ** 2
        )
        return sigmaLateral, sigmaAxial

    def calculateSurfacePoints(self, numPointsPerSection=10):
        surfacePointList = []
        X = self.X
        localTangents = self.localTangents
        PsiMax = self.calculateMaxViewAngle(X, localTangents)
        cameraNormals = self.calculateCameraNormals(X, localTangents)
        for i, x in enumerate(X):
            psiSteps = np.linspace(-PsiMax[i], PsiMax[i], numPointsPerSection)
            cameraNormal = cameraNormals[i, :]
            localTangent = localTangents[i, :]
            for psi in psiSteps:
                surfacePointList.append(
                    x
                    + self.radius
                    * (
                        np.cos(psi) * cameraNormal
                        + np.sin(psi) * (np.cross(cameraNormal, localTangent))
                    )
                )
        return np.array(surfacePointList)

    def calculatePointCloud(self, numPointsPerSection=10):
        noisySurfacePoints = []
        surfacePoints = self.calculateSurfacePoints(numPointsPerSection)
        SigmaLateral, SigmaAxial = self.calculateCameraNoise(surfacePoints)
        M = SigmaLateral.shape[0]
        for i, point in enumerate(surfacePoints):
            cov = self.camTransform[:3, :3].T @ np.diag(
                [SigmaLateral[i] ** 2, SigmaLateral[i] ** 2, SigmaAxial[i] ** 2]
            )
            noisySurfacePoints.append(np.random.multivariate_normal(point, cov, 1).T)

        # noisySurfacePoints = (
        #     surfacePoints
        #     + SigmaLateral[:, np.newaxis]
        #     * self.camTransform[:3, 0]
        #     * np.random.uniform(-1, 1, size=M)[:, np.newaxis]
        #     + SigmaLateral[:, np.newaxis]
        #     * self.camTransform[:3, 1]
        #     * np.random.uniform(-1, 1, size=M)[:, np.newaxis]
        #     + SigmaAxial[:, np.newaxis]
        #     * self.camTransform[:3, 2]
        #     * np.random.uniform(-1, 1, size=M)[:, np.newaxis]
        # )
        return np.array(noisySurfacePoints)
