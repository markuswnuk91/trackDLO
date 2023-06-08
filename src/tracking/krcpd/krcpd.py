from builtins import super
import os
import sys
import numpy as np
import numbers

try:
    sys.path.append(os.getcwd().replace("/src/tracking/krcpd", ""))
    from src.tracking.cpd.cpd import CoherentPointDrift
    from src.utils.utils import dampedPseudoInverse
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
        super().__init__(X=X, *args, **kwargs)

        self.q = qInit
        self.Dof = len(self.q)
        self.damping = 1 if damping is None else damping
        self.minDampingFactor = 1 if minDampingFactor is None else minDampingFactor
        self.dampingAnnealing = 0.97 if dampingAnnealing is None else dampingAnnealing
        self.ik_iterations = 1 if ik_iterations is None else ik_iterations

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

        X_desired = self.X + np.dot(self.G, self.W)
        self.X_desired = X_desired
        q = self.q
        ik_iterations = self.ik_iterations
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
        self.computeTargets()
        self.update_variance()

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
