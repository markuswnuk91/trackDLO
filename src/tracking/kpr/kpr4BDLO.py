import os
import sys
import numpy as np
import numbers
from warnings import warn

try:
    sys.path.append(os.getcwd().replace("/src/tracking/cpd", ""))
    from src.tracking.kpr.kpr import KinematicsPreservingRegistration
except:
    print("Imports for KPR failed.")
    raise


class KinematicsPreservingRegistration4BDLO(KinematicsPreservingRegistration):
    """Implementation of the kinematics preserving registration algorithm for branched deformable linear objects.

    Attributes:
    -------------
    for attributes see parent class. Here only differences to parent class are documented.

    model: BDLO model
        a model that provides Forward kinematics, Jacobians, and information about branch correspondance for each node.
        The provided model must provide the following two functions
            1) getPositions(q): returns a NxD np.array of spatial positions X from a (DoFx1) np.array of generalized coordinates q
            2) getJacobian(q,n): returns the jacobian towards the spatatial position X(n,:) from a (DoFyx1) np.array of generalized coordinates q and position information n
    """

    def __init__(self, B=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.B = [[x] for x in range(0, self.N)] if B is None else B
        self.K = len(self.B)
        self.PK = 1 / self.K * np.ones((self.N, self.M))
        self.PB = 1 / self.K * np.ones((self.K, 1))
        self.PNk = np.zeros((self.N, self.M))
        self.Nk = []
        for idxList in self.B:
            Nk = len(idxList)
            self.Nk.append(Nk)
            self.PNk[idxList, :] = 1 / Nk

    def estimateCorrespondance(self):
        """
        E-step: Compute the expectation step  of the EM algorithm.
        """
        P = np.sum((self.Y[None, :, :] - self.T[:, None, :]) ** 2, axis=2)

        c = (2 * np.pi * self.sigma2) ** (self.D / 2)
        c = c * self.mu / (1 - self.mu)
        c = c * self.N / self.M

        P = np.exp(-P / (2 * self.sigma2))
        Pmod = self.PK * self.PNk * P
        den = np.sum(Pmod, axis=0)
        den = np.tile(den, (self.N, 1))
        den[den == 0] = np.finfo(float).eps
        den += c

        self.Pden = den[0, :]
        self.P = np.divide(Pmod, den)
        self.Pt1 = np.sum(self.P, axis=0)
        self.P1 = np.sum(self.P, axis=1)
        self.Np = np.sum(self.P1)
        self.PY = np.matmul(self.P, self.Y)

    def updateParameters(self):
        """
        M-step: Calculate a new parameters of the registration.
        """

        jacobianDamping = (self.dampingAnnealing) ** (self.iteration) * self.damping
        if (
            jacobianDamping < self.minDampingFactor
        ):  # lower limit of regularization to ensure stability of matrix inversion
            jacobianDamping = self.minDampingFactor
        wStiffness = (self.stiffnessAnnelealing) ** (self.iteration) * self.wStiffness
        wGravity = (self.gravitationalAnnealing) ** (self.iteration) * self.wGravity
        A = np.zeros((self.Dof, self.Dof))
        B = np.zeros(self.Dof)
        stiffnessMatrix = (self.stiffnessAnnelealing) ** (
            self.iteration
        ) * self.stiffnessMatrix
        dEGrav = np.zeros(self.Dof)
        for n in range(0, self.N):
            J = self.model.getJacobian(self.q, n)
            dEGrav += self.gravity @ J
            for m in range(0, self.M):
                # A += self.P[n, m] * (self.Gq.T @ J.T @ J @ self.Gq)
                # B += self.P[n, m] * (self.Gq.T @ J.T @ (self.Y[m, :] - self.T[n, :]).T)
                A += self.wCorrespondance * self.P[n, m] * (J.T @ J)
                B += (
                    self.wCorrespondance
                    * self.P[n, m]
                    * (J.T @ (self.Y[m, :] - self.T[n, :]).T)
                )
        A += self.sigma2 * wStiffness * stiffnessMatrix
        B += (
            self.sigma2 * wStiffness * stiffnessMatrix @ (self.q0 - self.q)
        )  # add stiffness term for right side
        B += self.sigma2 * wGravity * dEGrav  # add gravitational term

        AInvDamped = self.dampedPseudoInverse(A, jacobianDamping)
        self.dq = AInvDamped @ B

        # update degrees of freedom
        self.updateDegreesOfFreedom()

        # set the new targets
        self.computeTargets()

        # update objective function
        Lold = self.L
        self.L = np.sum(np.log(self.Pden)) + self.D * self.M * np.log(self.sigma2) / 2

        self.diff = np.abs((self.L - Lold) / self.L)

        # update sigma
        yPy = np.dot(
            np.transpose(self.Pt1), np.sum(np.multiply(self.Y, self.Y), axis=1)
        )
        xPx = np.dot(np.transpose(self.P1), np.sum(np.multiply(self.T, self.T), axis=1))
        trPXY = np.sum(np.multiply(self.T, self.PY))
        self.sigma2 = (xPx - 2 * trPXY + yPy) / (self.Np * self.D)

        if self.sigma2 <= 0:
            self.sigma2 = self.tolerance / 10

        # update branch probability
        for i, idxList in enumerate(self.B):
            self.PB[i] = 1 / self.N * np.sum(self.P[idxList, :])
            self.PK[idxList, :] = self.PB[i]
