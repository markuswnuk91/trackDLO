import sys, os
import numpy as np
import numbers
from warnings import warn

try:
    sys.path.append(os.getcwd().replace("/src/modelling", ""))
    from src.simulation.dlo import DeformableLinearObject
except:
    print("Imports for discrete Model failed.")
    raise


class CameraModel(object):
    """Camera model, to create point cloud data from a DLO Model.

    Attributes
        ----------
        X: Nx3 np.array
            positions of the DLO representing its skeleton line expressed in some fixed reference coordinate system

        localTangent:
            Nx3 np.array
            unit vector representing the local tangent of the DLO's skeleton line.

        radius: float
            radius of the DLO.

        camTransform: 3x1 np.array
            Position of the camera in the reference coordinate system the DLO's positions are expressed
    """

    def __init__(self, camTransform=None, X=None, localTangents=None, radius = None, *args, **kwargs):
        
        if type(camTransform) is not np.ndarray or camTransform.ndim != 2:
            raise ValueError("The cam transform must be a 2D numpy array.")
        elif camTransform.shape != (4,4):
            raise ValueError("The cam transform must be a 4x4 homogenous matrix.")
        elif np.sum(np.linalg.inv(camTransform)@camTransform - np.eye(4)) >= 10e-5 or np.linalg.det(camTransform):
            raise ValueError("The cam transform must be a homogenous matrix. Obtained a error of {} between forward and inverse and a determinant of {}.".format(np.sum(np.linalg.inv(camTransform)@camTransform - np.eye(4)), np.linalg.det(camTransform)))

        if type(X) is not np.ndarray or X.ndim != 2:
            raise ValueError("The points on the skeleton line (X) must be a 2D numpy array.")
        elif X.shape[1] != 3:
            raise ValueError("The points on the skeleton line (X) must be specify xyz values for each point.")

        if type(localTangents) is not np.ndarray or localTangents.ndim != 2:
            raise ValueError("The local tangents to the skeleton line must be a 2D numpy array.")
        elif localTangents.shape[1] != 3 or localTangents.shape[0]!=X.shape[0]:
            raise ValueError("The local tangents to the skeleton line must be specify a 3 dimensional direction vector for each point in X.")
        elif np.any(np.linalg.norm(localTangents,axis=1)-1 >10e-8):
            warn(
                "Received non-normalized tangent vectors with lengths: {}. Normalizing the tangent vectors.".format(
                    np.linalg.norm(localTangents,axis=1)
                )
            )
            row_sums = localTangents.sum(axis=1)
            localTangents = localTangents / row_sums[:, np.newaxis]

        if radius is not None and (
            not isinstance(radius, numbers.Number) or radius < 0
        ):
            raise ValueError(
                "Expected a positive float for radius instead got: {}".format(
                    radius
                )
            )
        elif isinstance(radius, numbers.Number) and not isinstance(
            radius, float
        ):
            warn(
                "Received a non-float value for radius: {}. Casting to float.".format(
                    radius
                )
            )
            radius = float(radius)

        self.camTransform = np.identity(4) if camTransform is None else camTransform
        self.camPosition = camTransform[:3,3]
        self.X = X
        self.localTangents = localTangents


    def evalSkeletonLinePosition(self, s):
        return self.skeletonLineFun(s)

    def evalTangentVector(self, s):
        return self.localTangentFun(s)

    def evalSkeletonLinePositionInCameraCoordiates(self, s):
        return self.skeletonLineFun(s) - self.camPosition

    def evalEllipsisTiltAngle(self, s):
        x_cam = self.evalPositionInCameraCoordiates(self.skeletonLineFun(s))
        e_z = self.evalTangentVector(s)
        return np.arcsin(
            np.dot(x_cam, e_z) / (np.linalg.norm(x_cam) * np.linalg.norm(e_z))
        )

    def evalMaxTangentAngle(self, x_cam, e_z):
        tiltAngle = self.evalEllipsisTiltAngle(self, x_cam, e_z)
        return np.arccos(self.r / np.linalg.norm(x_cam) * np.cos(tiltAngle))

    def calcualteSurfacePoints(self, s, density=10):
        x_cam = self.evalSkeletonLinePositionInCameraCoordiates(s)
        e_z = self.evalTangentVector(s)
        theta = self.evalEllipsisTiltAngle(s)
        psi_max = self.evalMaxTangentAngel(s)
        psi = np.linspace(-psi_max, psi_max,density)
        c = - x_cam
        c_perp = c - np.dot(e_z,c)*e_z
        c_perp = 1/ np.linalg.norm(c_perp)* c_perp

        # pointCloud = self.evalSkeletonLinePositionInCameraCoordiates(s) - (
        #     self.radius + np.tan(self.evalEllipsisTiltAngle(s))
        # ) * np.linalg.norm(self.evalSkeletonLinePositionInCameraCoordiates(s))

        self.radius * c_perp
        + self.radius * np.cos(psi) np.tan(theta) * e_z
        + self.radius * np.sin(psi) * np.cross(x_cam, e_z)

        return surfacePoints
