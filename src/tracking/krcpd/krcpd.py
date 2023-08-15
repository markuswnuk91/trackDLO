from builtins import super
import os
import sys
import numpy as np
import numbers

try:
    sys.path.append(os.getcwd().replace("/src/tracking/krcpd", ""))
    from src.tracking.cpd.cpd import CoherentPointDrift
    from src.utils.utils import dampedPseudoInverse
    from src.utils.utils import gaussian_kernel
except:
    print("Imports for KR-CPD failed.")
    raise


class KinematicRegularizedCoherentPointDrift(CoherentPointDrift):
    def __init__(
        self,
        model,
        qInit,
        damping=None,
        minDampingFactor=None,
        dampingAnnealing=None,
        ik_iterations=None,
        *args,
        **kwargs
    ):
        self.model = model
        self.qInit = qInit
        X = self.model.getPositions(self.qInit)
        self.Xreg = X
        self.q = qInit
        self.Dof = len(self.q)
        super().__init__(X=X, *args, **kwargs)
        self.initializeKinematicRegularizationParameters(
            damping=damping,
            minDampingFactor=minDampingFactor,
            dampingAnnealing=dampingAnnealing,
            ik_iterations=ik_iterations,
        )

    def initializeKinematicRegularizationParameters(
        self,
        damping=None,
        minDampingFactor=None,
        dampingAnnealing=None,
        ik_iterations=None,
    ):
        super().initializeParameters(self.alpha, self.beta)
        self.damping = 1 if damping is None else damping
        self.minDampingFactor = 1 if minDampingFactor is None else minDampingFactor
        self.dampingAnnealing = 0.97 if dampingAnnealing is None else dampingAnnealing
        self.ik_iterations = 1 if ik_iterations is None else ik_iterations
        return

    def reinitializeParameters(self):
        self.initializeWeights()
        self.initializeCorrespondances()
        self.estimateCorrespondance()
        self.update_variance()
        return

    def estimateCorrespondance(self):
        """
        E-step: Compute the expectation step  of the EM algorithm.
        """
        if self.normalize:
            # normalize to 0 mean
            Y_hat = self.Y - np.mean(self.Y)
            Xreg_hat = self.Xreg - np.mean(self.Xreg)
            # normalize to 0 variance
            scalingFactor_Xreg = np.sqrt(np.sum(self.Xreg**2) / self.N)
            scalingFactor_Y = np.sqrt(np.sum(self.Y**2) / self.M)
            Y_hat = Y_hat / scalingFactor_Y
            Xreg_hat = Xreg_hat / scalingFactor_Xreg
            P = np.sum((Y_hat[None, :, :] - Xreg_hat[:, None, :]) ** 2, axis=2)

            c = (2 * np.pi * self.sigma2) ** (self.D / 2)
            c = c * self.mu / (1 - self.mu)
            c = c * self.N / self.M

            P = np.exp(-P / (2 * self.sigma2))
            den = np.sum(P, axis=0)
            den = np.tile(den, (self.N, 1))
            den[den == 0] = np.finfo(float).eps
            den += c

            self.Pden = den[0, :]
            self.P = np.divide(P, self.Pden)
            self.Pt1 = np.sum(self.P, axis=0)
            self.P1 = np.sum(self.P, axis=1)
            self.Np = np.sum(self.P1)
            self.PY = np.matmul(self.P, self.Y)
        else:
            P = np.sum((self.Y[None, :, :] - self.Xreg[:, None, :]) ** 2, axis=2)

            c = (2 * np.pi * self.sigma2) ** (self.D / 2)
            c = c * self.mu / (1 - self.mu)
            c = c * self.N / self.M

            P = np.exp(-P / (2 * self.sigma2))
            den = np.sum(P, axis=0)
            den = np.tile(den, (self.N, 1))
            den[den == 0] = np.finfo(float).eps
            den += c

            self.Pden = den[0, :]
            self.P = np.divide(P, self.Pden)
            self.Pt1 = np.sum(self.P, axis=0)
            self.P1 = np.sum(self.P, axis=1)
            self.Np = np.sum(self.P1)
            self.PY = np.matmul(self.P, self.Y)

    def updateParameters(self):
        """
        M-step: Calculate a new estimate of the deformable transformation.
        See Eq. 22 of https://arxiv.org/pdf/0905.2635.pdf.
        """
        if self.low_rank is False:
            A = np.dot(np.diag(self.P1), self.G) + self.alpha * self.sigma2 * np.eye(
                self.N
            )
            B = self.PY - np.dot(np.diag(self.P1), self.X)
            self.W = np.linalg.solve(A, B)

        elif self.low_rank is True:
            # Matlab code equivalent can be found here:
            # https://github.com/markeroon/matlab-computer-vision-routines/tree/master/third_party/CoherentPointDrift
            dP = np.diag(self.P1)
            dPQ = np.matmul(dP, self.Q)
            F = self.PY - np.matmul(dP, self.X)

            self.W = (
                1
                / (self.alpha * self.sigma2)
                * (
                    F
                    - np.matmul(
                        dPQ,
                        (
                            np.linalg.solve(
                                (
                                    self.alpha * self.sigma2 * self.inv_S
                                    + np.matmul(self.Q.T, dPQ)
                                ),
                                (np.matmul(self.Q.T, F)),
                            )
                        ),
                    )
                )
            )
            QtW = np.matmul(self.Q.T, self.W)
            self.E = self.E + self.alpha / 2 * np.trace(
                np.matmul(QtW.T, np.matmul(self.S, QtW))
            )

        # kinematic regularization
        jacobianDamping = (self.dampingAnnealing) ** (self.iteration) * self.damping
        dq = np.zeros(len(self.q))
        q = self.q
        ik_iterations = self.ik_iterations
        X_desired = self.X + np.dot(self.G, self.W)
        for i in range(0, ik_iterations):
            X_current = self.model.getPositions(q)
            X_error = X_desired - X_current
            jacobians = []
            for n in range(0, self.N):
                J_hat = self.model.getJacobian(q, n)
                jacobians.append(J_hat)
            J = np.vstack(jacobians)
            dq = dampedPseudoInverse(J, jacobianDamping) @ X_error.flatten()
            q = q + dq
        # update generalized coordinates
        self.q = q
        self.computeRegularizedConfiguration()
        # self.W = (1 - (self.P1 / np.max(self.P1)))[:, None] * np.linalg.inv(self.G) @ (
        #     self.Xreg - self.X
        # ) + (self.P1 / np.max(self.P1))[:, None] * self.W

        self.computeTargets()

        self.update_variance()

    def computeRegularizedConfiguration(self, q=None):
        if q is not None:
            Xreg = self.model.getPositions(q)
            return Xreg
        else:
            self.Xreg = self.model.getPositions(self.q)
            return

    def computeTargets(self, X=None):
        """
        Update the targets using the new estimate of the parameters.
        Attributes
        ----------
        X: numpy array, optional
            Array of points to transform - use to predict on new set of points.
            Best for predicting on new points not used to run initial registration.
                If None, self.X used.

        Returns
        -------
        If X is None, returns None.
        Otherwise, returns the transformed X.

        """
        if X is not None:
            G = gaussian_kernel(X=X, beta=self.beta, Y=self.X)
            return X + np.dot(G, self.W)
        else:
            if self.low_rank is False:
                self.T = self.X + np.dot(self.G, self.W)
                # if np.any(self.P1 > 0):
                #     self.T = self.Xreg + np.exp(-(1 - (self.P1 / np.max(self.P1))))[
                #         :, None
                #     ] * (self.X + np.dot(self.G, self.W) - self.Xreg)
                # else:
                #     self.T = self.Xreg
            elif self.low_rank is True:
                self.T = self.X + np.matmul(
                    self.Q, np.matmul(self.S, np.matmul(self.Q.T, self.W))
                )
                return
