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
        Placeholder for child classes.s
        """
        self.skel.setPositions(q)
        J = np.zeros((3, self.Dof))
        dartJacobian = self.skel.getBodyNode(n).getWorldJacobian(np.array([0, 0, 0]))[
            3:6, :
        ]
        if dartJacobian.shape[1] < self.Dof:
            J = np.pad(
                dartJacobian,
                ((0, 0), (0, self.Dof - dartJacobian.shape[1] % self.Dof)),
                "constant",
            )
        elif dartJacobian.shape[1] == self.Dof:
            J = dartJacobian
        else:
            raise ValueError("Dimension of Jacobian seems wrong.")
        return J
