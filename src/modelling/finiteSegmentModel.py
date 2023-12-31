import sys, os
from builtins import super
import numpy as np
import dartpy as dart
import numbers
from warnings import warn
from scipy.spatial.transform import Rotation as R

try:
    sys.path.append(os.getcwd().replace("/src/modelling", ""))
    from src.simulation.dlo import DeformableLinearObject
except:
    print("Imports for discrete Model failed.")
    raise


class FiniteSegmentModel(DeformableLinearObject):
    """Implementation of a finite segment representation of a DLO.
    The implementation is based on the theoretic principles of Discrete Kirchoff Rods from the paper:
    "Miklos Bergou et al., Discrete Elastic Rods, ACM Transactions on Graphics (SIGGRAPH), 2008".
    It uses the Dynamics Animation and Robotics Toolkit (DART) to model the underlying kinematics.
    For human-readability the user interface uses extrinsic euler angles to represent the rotations inbetween individual segments (Kinematic Chain Representation), instead of DART's logMap representation.

    This class contains the functionality to map between the different representations.
    KinematicChain <--> DART <--> Discrete Kirchoff Rod
    Furthermore it provides functionality to map from the continous space (local coordinate s) to the discrete space (segments i).

    Attributes
        ----------
        N: int
            Number of segments

        Discrete Kirchhoff Rod (Minimal coordinate representaion as described in Bergou et al.):

            phis: np.array
                Bending angles around the curvature binormal for each segment as described in Bergou et al.

            thetas: np.array
                Torsion angles around the normal for each segment as described in Bergou et al.

            tangents: np.array
                tangent vectors pointing in the direction of the segment as described in Bergou et al. (equivalent with the z-coodinate of the dart models local bodyNode coordinate system)

            curvatureBinormals: np.array
                curvature binormals as described in Bergou et al.

        Kinematic Chain Representation (extrinsic Euler Angles)
        alphas: np.array
            Rotation angles around the local x-coordinates for each bodyNode coordinate system in extrinsic euler angle representation (rotation around fixed parent coordinate system)

        betas: np.array
            Rotation angles around the local y-coordinates for each bodyNode coordinate system in extrinsic euler angle representation (rotation around fixed parent coordinate system)

        gammas: np.array
            Rotation angles around the local z-coordinates for each bodyNode coordinate system in extrinsic euler angle representation (rotation around fixed parent coordinate system)

        x0: np.array
            1x3 position of the beginning of the first segment

        rot0: np.array
            1x3 Rotation angles to transform from the reference coordinate system to the first segment in extrinsic euler angle representation (rotation around fixed parent coordinate system)

        Dart Representation
        q: np.array
            Degrees of freedom of the dart representation where q[0:3] are the rotations of the first dart.dynamics.FreeJoint, q[3:6] are the x,y,z translations of the first dart.dynamicsFreeJoint, q[6:] are the degrees of freedom of the remaining dart.dynamics.BallJoint. DART requires the angles to be represented in logMap (rotVec) representation.
    """

    def __init__(
        self,
        L=None,
        N=None,
        Rflex=None,
        Rtor=None,
        Roh=None,
        x0=None,
        rot0=None,
        alphas=None,
        betas=None,
        gammas=None,
        gravity=None,
        *args,
        **kwargs
    ):

        if L is not None and (not isinstance(L, numbers.Number) or L < 0):
            raise ValueError(
                "Expected a positive float for length of the DLO instead got: {}".format(
                    L
                )
            )

        if N is not None and (not isinstance(N, numbers.Number) or N < 1):
            raise ValueError(
                "Expected a positive integer of at least 1 for the number of segments instead got: {}".format(
                    N
                )
            )
        elif isinstance(N, numbers.Number) and not isinstance(N, int):
            warn(
                "Received a non-integer value for number of segments: {}. Casting to integer.".format(
                    N
                )
            )
            N = int(N)

        self.L = 1 if L is None else L
        self.N = 10 if N is None else N
        self.numDofs = 3 * self.N + 3
        self.Rflex = 1 if Rflex is None else Rflex
        self.Rtor = 1 if Rtor is None else Rtor
        self.Roh = 0.1 if Roh is None else Roh
        self.x0 = np.zeros(3) if x0 is None else x0
        self.rot0 = np.zeros(3) if rot0 is None else rot0
        self.alphas = np.zeros(self.N - 1) if alphas is None else alphas
        self.betas = np.zeros(self.N - 1) if betas is None else betas
        self.gammas = np.zeros(self.N - 1) if gammas is None else gammas
        self.gravity = np.array([0, 0, 9.81]) if gravity is None else gravity
        self.q = self.mapAnglesToDartPositions(
            self.x0, self.rot0, self.alphas, self.betas, self.gammas
        )
        # self.phis = np.zeros(self.N - 1) if phis is None else phis
        # self.thetas = np.zeros(self.N - 1) if thetas is None else thetas

        # initialize dart model
        super().__init__(
            numSegments=self.N,
            length=self.L,
            *args,
            **kwargs,
        )

    def mapDartPositionsToAngles(self, q):
        x0 = q[3:6]
        # q[:3]
        rot0 = R.from_rotvec([q[0], q[1], q[2]]).as_euler("xyz")
        # alphas = q[6::3]
        # betas = q[7::3]
        # gammas = q[8::3]
        angles = R.from_rotvec(np.reshape(q[6:], (-1, 3))).as_euler("xyz")
        alphas = angles[:, 0]
        betas = angles[:, 1]
        gammas = angles[:, 2]
        return x0, rot0, alphas, betas, gammas

    def mapAnglesToDartPositions(self, x0, rot0, alphas, betas, gammas):
        q = np.zeros(self.numDofs)
        q[3:6] = x0
        # q[:3] = rot0
        q[:3] = R.from_euler("xyz", rot0).as_rotvec()
        if self.N > 1:
            rotVecs = R.from_euler(
                "xyz", np.column_stack((alphas, betas, gammas))
            ).as_rotvec()
            q[6:] = rotVecs.flatten()
            # q[6::3] = alphas
            # q[7::3] = betas
            # q[8::3] = gammas
        return q

    def convertPhiToBallJointPositions(self, phi: float, axis: np.array):
        """Converts the minimal coordinate formulation from Bergou et al. to Dart Balljoint Positions.

        For an example the conversion see DART Tutorial https://dartsim.github.io/tutorials_collisions.html

        Args:
            phi (float): rotation angle around the curvature binormal
            axis (np.array): curvature binormal

        Returns:
            (np.array): the dart DOF positions of the balljoint . These are the rotation angles alpha, beta, gamma around the parent coodinate frame's  x,y,z, axis.
        """
        rotMat = R.from_rotvec(phi * axis).as_matrix()
        return dart.dynamics.BallJoint.convertToPositions(rotMat)

    def getJointLocalCoordinates(self):
        return np.insert(np.cumsum(np.array(self.segmentLengths)) / self.length, 0, 0)

    def getBodyNodeIndexFromLocalCoodinate(self, s: float):
        """returns the bodyNode index corresponding to a local coordinate

        Args:
            s (float): local coordinate in [0,1]

        Returns:
            int: bodyNode index of the body the local coordinate corresponds to.
        """
        if 1 - s <= np.finfo(float).eps:
            return self.skel.getNumBodyNodes() - 1
        elif s <= np.finfo(float).eps:
            return 0
        else:
            jointLocalCoordinates = self.getJointLocalCoordinates()
            return (
                next(
                    index[0]
                    for index in enumerate(jointLocalCoordinates)
                    if index[1] > s
                )
                - 1
            )

    def getOffsetInBodyNodeCoordinatesFromLocalCoordiate(self, bodyNodeIndex, s: float):
        """returns the offset from a center of a bodyNode to the location corresponding to a local coordinate.
        The offset is expressed in the cooresponding local bodyNode frame

        Args:
            s (float): local coordinate in [0,1]

        Returns:
            np.array: 3x1 offset vector fom the body node center to the position corresponding to the given local coordinate.
        """
        bodyNodeIndex = self.getBodyNodeIndexFromLocalCoodinate(s)
        localCoordsJoints = self.getJointLocalCoordinates()
        sLower = localCoordsJoints[bodyNodeIndex]
        sUpper = localCoordsJoints[bodyNodeIndex + 1]
        sCenter = sLower + (sUpper - sLower) / 2
        sOffset = (s - sCenter) / (sUpper - sLower)
        offset = np.array([0, 0, sOffset * self.segmentLengths[bodyNodeIndex]])
        return offset

    def getCartesianPositionFromLocalCoordinate(self, s: float):
        if s <= np.finfo(float).eps:
            return self.getCartesianPositionSegmentStart(0)
        elif 1 - s <= np.finfo(float).eps:
            return self.getCartesianPositionSegmentEnd(-1)
        else:
            correspondBodyNodeIdx = self.getBodyNodeIndexFromLocalCoodinate(s)
            offset = self.getOffsetInBodyNodeCoordinatesFromLocalCoordiate(
                correspondBodyNodeIdx, s
            )
            return self.getCartesianPositionSegmentWithOffset(
                correspondBodyNodeIdx, offset
            )

    def getCaresianPositionsFromLocalCoordinates(self, S: np.array):
        """returns cartesian positions for all local coordinates in S


        Args:
            S (np.array): Nx1 array of local coordinates in [0,1]

        Return:
            X (np.array): Nx3 array of positions corresponding to the local coordinates
        """
        X = np.zeros((S.size, 3))
        for i, s in enumerate(S):
            X[i, :] = self.getCartesianPositionFromLocalCoordinate(s)
        return X

    def getCartesianPositionSegmentCenter(self, bodyNodeIndex: int):
        """returns the cartesian position of the center of a segment

        Args:
            bodyNodeIndex (int): the index of the bodyNode

        Returns:
            np.array: cartesian position of the center of the segement
        """
        return self.skel.getBodyNode(bodyNodeIndex).getWorldTransform().translation()

    def getCartesianPositionSegmentWithOffset(
        self, bodyNodeIndex: int, offset: np.array
    ):
        """returns the cartesian position of the center of a segment

        Args:
            bodyNodeIndex (int): the index of the bodyNode

        Returns:
            np.array: cartesian position of the center of the segement
        """
        return (
            self.skel.getBodyNode(bodyNodeIndex).getWorldTransform().matrix()
            @ np.append(offset, 1)
        )[:3]

    def getCartesianPositionSegmentStart(self, bodyNodeIndex: int):
        """returns the cartesian position of the beginning of a segment

        Args:
            bodyNodeIndex (int): the index of the bodyNode

        Returns:
            np.array: cartesian position of the start of the segement
        """
        bodyNodeTransform = (
            self.skel.getBodyNode(bodyNodeIndex).getWorldTransform().matrix()
        )
        relativeTransformToParentJoint = (
            self.skel.getBodyNode(bodyNodeIndex)
            .getParentJoint()
            .getTransformFromChildBodyNode()
            .matrix()
        )
        return (bodyNodeTransform @ relativeTransformToParentJoint)[:3, 3]

    def getCartesianPositionSegmentEnd(self, bodyNodeIndex: int):
        """returns the cartesian position of the beginning of a segment

        Args:
            bodyNodeIndex (int): the index of the bodyNode

        Returns:
            np.array: cartesian position of the start of the segement
        """
        if bodyNodeIndex == -1:
            bodyNodeIndex = self.skel.getNumBodyNodes() - 1

        bodyNodeTransform = (
            self.skel.getBodyNode(bodyNodeIndex).getWorldTransform().matrix()
        )
        relativeTransformToParentJoint = (
            self.skel.getBodyNode(bodyNodeIndex)
            .getParentJoint()
            .getTransformFromChildBodyNode()
            .matrix()
        )
        # reverse direction to go to end of segment
        relativeTransformToParentJoint[:3, 3] = -relativeTransformToParentJoint[:3, 3]
        return (bodyNodeTransform @ relativeTransformToParentJoint)[:3, 3]

    def getJacobianFromLocalCoordinate(self, s: float):
        correspondingBodyNode = self.getBodyNodeIndexFromLocalCoodinate(s)
        offset = self.getOffsetInBodyNodeCoordinatesFromLocalCoordiate(
            correspondingBodyNode, s
        )
        return (
            self.skel.getBodyNode(correspondingBodyNode).getWorldJacobian(offset).copy()
        )

    def getJacobianFromLocalCoordinates(self, S: np.array):
        """returns jacobians for all local coordinates in S
        Args:
            S (np.array): Nx1 array of local coordinates in [0,1]

        Return:
            jacobianList (list of np.arrays): list of jacobians corresponding to the local coordinates
        """
        jacobianList = []
        for s in S:
            J = self.getJacobianFromLocalCoordinate(s)
            jacobianList.append(J)
        return jacobianList

    def getCartesianPositionRootJoint(self):
        return self.getCartesianPositionSegmentStart(0)
