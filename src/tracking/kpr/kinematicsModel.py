import os
import sys
import numpy as np
import numbers
from warnings import warn
import dartpy as dart


class KinematicsModelDart(object):
    def __init__(self, dartSkel, *args, **kwargs):
        self.skel = dartSkel
        self.N = self.skel.getNumBodyNodes()
        self.Dof = self.skel.getNumDofs()

    def getPositions(self, q):
        """
        Placeholder for child classes.
        """
        self.skel.setPositions(q)
        X = np.zeros((self.N, 3))
        for n in range(0, self.skel.getNumBodyNodes()):
            X[n, :] = self.skel.getBodyNode(n).getWorldTransform().translation()
        return X

    def getJacobian(self, q, n, method=None):
        """
        Placeholder for child classes
        """
        self.skel.setPositions(q)
        if method is None:
            # J = np.zeros((3, self.Dof))

            # dartJacobian = self.skel.getBodyNode(n).getWorldJacobian(np.array([0, 0, 0]))[
            #     3:6, :
            # ]
            # # dartJacobian[:, 3:6] = (
            # #     np.linalg.inv(self.skel.getBodyNode(0).getWorldTransform().rotation())
            # #     @ dartJacobian[:, 3:6]
            # # )

            # # (
            # #     np.linalg.inv(self.skel.getBodyNode(0).getWorldTransform().rotation())
            # #     @ self.skel.getBodyNode(n).getWorldJacobian(np.array([0, 0, 0]))[3:6, :]
            # # )
            # if dartJacobian.shape[1] < self.Dof:
            #     #     J = np.pad(
            #     #         dartJacobian,
            #     #         ((0, 0), (0, self.Dof - dartJacobian.shape[1] % self.Dof)),
            #     #         "constant",
            #     #     )
            #     indexPointer = 0
            #     paddedJacobian = np.zeros((3, self.skel.getNumDofs()))
            #     for i in range(0, self.skel.getNumDofs()):
            #         if self.skel.getBodyNode(n).dependsOn(i):
            #             paddedJacobian[:, i] = dartJacobian[:, indexPointer]
            #             indexPointer += 1
            #     J = paddedJacobian
            # elif dartJacobian.shape[1] == self.Dof:
            #     J = dartJacobian
            # else:
            #     raise ValueError("Dimension of Jacobian seems wrong.")

            # return self.skel.getWorldJacobian(self.skel.getBodyNode(n))[3:, :]

            # Darts' world jacobian seems wrong for free floating base (translational degrees of freedom are not identity matrix)
            return np.linalg.inv(
                self.skel.getBodyNode(0).getTransform().rotation()
            ) @ self.skel.getLinearJacobian(self.skel.getBodyNode(n))
            # jacobian = self.skel.getLinearJacobian(self.skel.getBodyNode(n))
            # jacobian[:, :3] = np.eye(3)
            # return self.skel.getLinearJacobian(self.skel.getBodyNode(n))
            # Compute adjoint of transform from body to world

        elif method == "world":
            if n == 1:
                joint_offset = - self.skel.getBodyNode(n).getChildJoint(0).getTransformFromParentBodyNode().translation()
            else:
                joint_offset = self.skel.getBodyNode(n).getParentJoint().getTransformFromChildBodyNode().translation()

            return self.skel.getWorldJacobian(self.skel.getBodyNode(n),joint_offset)[3:,:]
        elif method == "first_body":
            return self.skel.getJacobian(self.skel.getBodyNode(n),self.skel.getBodyNode(0))[3:,:]
        elif method == "adjoint":
            raise NotImplementedError
            self.skel.setPositions(q)
            # Get the body node's transform in world coordinates
            T = self.skel.getBodyNode(n).getWorldTransform()

            joint_offset = self.skel.getBodyNode(n).getChildJoint(0).getTransformFromParentBodyNode().translation()

            # Get the spatial Jacobian in the body frame (or local frame at offset)
            J_body = self.skel.getBodyNode(n).getJacobian(-joint_offset)

            # Compute adjoint of transform from body to world
            R = T.rotation()
            p = T.translation()
            px = self.skew_symmetric(p)
            Ad_T = np.block([
                [R, np.zeros((3, 3))],
                [px @ R, R]
            ])

            # Transform to world frame using adjoint
            J_world = Ad_T @ J_body
            return J_world[3:, :]  # Return only the linear part

    def dependsOn(self, bodyNodeIndex, genCoordinateIndex):
        return self.skel.getBodyNode(bodyNodeIndex).dependsOn(genCoordinateIndex)
    
    def skew_symmetric(self,v):
        """Create a 3x3 skew-symmetric matrix from a 3D vector."""
        screw_symmetric_matrix = np.array([
            [0,     -v[2],  v[1]],
            [v[2],   0,    -v[0]],
            [-v[1],  v[0],  0]
        ])
        return screw_symmetric_matrix
