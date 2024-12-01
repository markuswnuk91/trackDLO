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

    def getJacobian(self, q, n):
        """
        Placeholder for child classes
        """
        self.skel.setPositions(q)
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

    def dependsOn(self, bodyNodeIndex, genCoordinateIndex):
        return self.skel.getBodyNode(bodyNodeIndex).dependsOn(genCoordinateIndex)
