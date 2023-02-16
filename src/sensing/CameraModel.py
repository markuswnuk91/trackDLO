import sys, os
import numpy as np
import numbers
from warnings import warn

try:
    sys.path.append(os.getcwd().replace("/src/sensing", ""))
except:
    print("Imports for CameraModel failed.")
    raise


class CameraModel(object):
    """Camera model, to create point cloud data from a DLO Model.

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
