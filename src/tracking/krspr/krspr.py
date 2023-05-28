from builtins import super
import os
import sys
import numpy as np
import numbers

try:
    sys.path.append(os.getcwd().replace("/src/tracking/spr", ""))
    from src.tracking.spr.spr import StructurePreservedRegistration
except:
    print("Imports for SPR failed.")
    raise


class KinematicRegularizedStructurePreservedRegistration(
    StructurePreservedRegistration
):
    def __init__(
        self,
        model,
        qInit,
        damping=None,
        minDampingFactor=None,
        dampingAnnealing=None,
        *args,
        **kwargs
    ):
        self.model = model
        self.qInit = qInit
        X = self.model.getPositions(self.qInit)
        super().__init__(X=X, *args, **kwargs)

        self.q = qInit
        self.Dof = len(self.q)
        self.damping = 1 if damping is None else damping
        self.minDampingFactor = 1 if minDampingFactor is None else minDampingFactor
        self.dampingAnnealing = 0.97 if dampingAnnealing is None else dampingAnnealing

    def computeTargets(self, q=None):
        """
        Update the targets using the new estimate of the parameters.
        Attributes
        ----------
        q: numpy array, optional
            Array of joint positions of the kinematis model

        Returns
        -------
        If X is None, returns None.
        Otherwise, returns the transformed X.

        """
        if q is not None:
            T = self.model.getPositions(q)
            return T
        else:
            self.T = self.model.getPositions(self.q)
            return

    def updateParameters(self):
        """
        M-step: Calculate the new parameters of the registration.
        """

        tauFactor = (self.tauAnnealing) ** (self.iteration) * self.tauFactor
        lambdaFactor = (self.lambdaAnnealing) ** (self.iteration) * self.lambdaFactor
        jacobianDamping = (self.dampingAnnealing) ** (self.iteration) * self.damping
        if (
            jacobianDamping < self.minDampingFactor
        ):  # lower limit of regularization to ensure stability of matrix inversion
            jacobianDamping = self.minDampingFactor

        dP1 = np.diag(self.P1)
        A = (
            np.dot(dP1, self.G)
            + lambdaFactor * self.sigma2 * np.eye(self.N)
            + tauFactor * self.sigma2 * np.dot(self.Phi, self.G)
        )
        B = (
            self.PY
            - np.dot(dP1, self.X)
            - tauFactor * self.sigma2 * np.dot(self.Phi, self.X)
        )
        self.W = np.linalg.solve(A, B)

        # rotations and translation split
        # # update kinematics model
        # dq = np.zeros(len(self.q))
        # # compute differences
        # dX = (self.X + np.dot(self.G, self.W)) - self.T
        # # update translation
        # self.q[3:6] = self.q[3:6] + dX[0, :]
        # # compute rotations (solve IK)
        # X_desired = self.X + np.dot(self.G, self.W)
        # self.X_desired = X_desired
        # q = self.q
        # ik_iterations = 20
        # for i in range(0, ik_iterations):
        #     X_current = self.model.getPositions(q)
        #     X_error = X_desired - X_current
        #     jacobians = []
        #     for n in range(0, self.N):
        #         J_hat = self.model.getJacobian(q, n)
        #         jacobians.append(
        #             np.hstack((J_hat[:, :3], J_hat[:, 6:]))
        #         )  # only for rotational dofs
        #     J = np.vstack(jacobians)
        #     dq_rot = self.dampedPseudoInverse(J, jacobianDamping) @ X_error.flatten()
        #     dq[:3] = dq_rot[:3]
        #     dq[6:] = dq_rot[3:]
        #     q = q + dq

        dq = np.zeros(len(self.q))
        dX = (self.X + np.dot(self.G, self.W)) - self.T
        X_desired = self.X + np.dot(self.G, self.W)
        self.X_desired = X_desired
        q = self.q
        ik_iterations = 20
        for i in range(0, ik_iterations):
            X_current = self.model.getPositions(q)
            X_error = X_desired - X_current
            jacobians = []
            for n in range(0, self.N):
                J_hat = self.model.getJacobian(q, n)
                jacobians.append(J_hat)
            J = np.vstack(jacobians)
            dq = self.dampedPseudoInverse(J, jacobianDamping) @ X_error.flatten()
            q = q + dq

        # update generalized coordinates
        self.q = q
        # set the new targets
        self.computeTargets()

        # update objective function
        Lold = self.L
        self.L = (
            np.sum(np.log(self.Pden))
            + self.D * self.M * np.log(self.sigma2) / 2
            - lambdaFactor / 2 * np.trace(np.transpose(self.W) @ self.G @ self.W)
            - tauFactor / 2 * np.trace(np.transpose(self.T) @ self.Phi @ self.T)
        )

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

    def dampedPseudoInverse(self, J, dampingFactor):
        dim = J.shape[0]
        dampedPseudoInverse = J.T @ np.linalg.inv(
            J @ J.T + dampingFactor**2 * np.eye(dim)
        )
        return dampedPseudoInverse
