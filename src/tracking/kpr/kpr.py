import os
import sys
import numpy as np
import numbers
from warnings import warn

try:
    sys.path.append(os.getcwd().replace("/src/tracking/kpr", ""))
    from src.tracking.registration import NonRigidRegistration
    from src.tracking.kpr.kinematicsModel import KinematicsModelDart
    from src.utils.utils import dampedPseudoInverse
except:
    print("Imports for KPR failed.")
    raise


class KinematicsPreservingRegistration(NonRigidRegistration):
    """Base class for kinematics based non-rigid registration algorithm.

    Attributes:
    -------------
    q: numpy array
        Nx1 array of degrees of freedom

    qInit: numpy array
        Nx1 array of initial values for the degrees of freedom

    q0: numpy array
        Nx1 array of rest positions for the degrees of freedom to which the model should converge if no target points are given

    q_dot: numpy array
        Nx1 array of velocities (change per iteration) of the degrees of freedom

    Y: numpy array
        MxD array of input target points (e.g. point cloud data)

    T: numpy array
        NxD array of output targets that should approximate the input target points

    N: int
        Number of degrees of freedom

    M: int
        Number of data points

    D: int
        Dimensionality target points and targets

    model: kinematic model class
        a kinematic model that provides functionality to calculate positions X and Jacobians J from gernalized coordinates q.
        The provided model must provide the following two functions
            1) self.getPositions(q): returns a NxD np.array of spatial positions X from a (DoFx1) np.array of generalized coordinates q
            2) self.getJacobian(q,n): returns the jacobian towards the spatatial position X(n,:) from a (DoFyx1) np.array of generalized coordinates q and position information n

    iterations: int
        The current iteration throughout the registration

    max_iterations: int
        Maximum number of iterations the registration performs before terminating

    tolerance: float (positive)
        tolerance for checking convergence.
        Registration will terminate once the difference between
        consecutive objective function values falls within this tolerance.

    diff: float (positive)
        The absolute normalized difference between the current and previous objective function values.

    L: float
        The log-likelyhood of the dataset probability given the parameterization. SPR aims to update the parameters such that they maximize this value.

    P: numpy array
        MxN array of probabilities.
        P[m, n] represents the probability that the m-th target point
        corresponds to the n-th target.

    mu: float (between 0 and 1)
        Contribution of the uniform distribution to account for outliers.
        Valid values span 0 (inclusive) and 1 (exclusive).
    """

    def __init__(
        self,
        qInit,
        model: KinematicsModelDart,
        ik_iterations=None,
        damping=None,
        dampingAnnealing=None,
        minDampingFactor=None,
        *args,
        **kwargs
    ):
        X = model.getPositions(qInit)
        super().__init__(X=X, *args, **kwargs)
        if type(qInit) is not np.ndarray or qInit.ndim > 1:
            raise ValueError("The degrees of freedom (q) must be a 1D numpy array.")

        if ik_iterations is not None and (
            not isinstance(ik_iterations, numbers.Number) or ik_iterations < 0
        ):
            raise ValueError(
                "Expected a positive integer for max_iterations instead got: {}".format(
                    ik_iterations
                )
            )
        elif isinstance(ik_iterations, numbers.Number) and not isinstance(
            ik_iterations, int
        ):
            warn(
                "Received a non-integer value for max_iterations: {}. Casting to integer.".format(
                    ik_iterations
                )
            )
            ik_iterations = int(ik_iterations)

        self.qInit = qInit
        self.q = qInit.copy()
        self.model = model
        self.Dof = self.q.size
        self.ik_iterations = 1 if ik_iterations is None else ik_iterations

        self.damping = 1 if damping is None else damping
        self.dampingAnnealing = 0.97 if dampingAnnealing is None else dampingAnnealing
        self.minDampingFactor = 1 if minDampingFactor is None else minDampingFactor

        self.diff = np.inf
        self.L = -np.inf
        self.P = np.zeros((self.N, self.M))
        self.Pden = np.zeros((self.M))
        self.Pt1 = np.zeros((self.M,))
        self.P1 = np.zeros((self.N,))
        self.Np = 0
        self.PY = np.zeros((self.N, self.D))
        self.W = np.zeros((self.Dof, self.D))

    def isConverged(self):
        """
        Checks if change of cost function is below the defined tolerance
        """
        return self.diff < self.tolerance

    def updateDegreesOfFreedom(self):
        self.q += self.dq

    def computeTargets(self, q=None):
        """
        Update the targets using the new estimate of the parameters.
        Attributes
        ----------
        q: numpy array, optional
            Array of points to transform - use to predict a new set of points.
            Best for predicting on new points not used to run initial registration.
                If None, self.q used.

        Returns
        -------
        T: numpy array
            the transformed targets T(X).

        """
        if q is not None:
            T = np.zeros((self.N, self.D))
            for n in range(0, self.N):
                T[n, :] = self.model.getPositions(q)[n, :]
            return T
        else:
            for n in range(0, self.N):
                self.T[n, :] = self.model.getPositions(self.q)[n, :]
        return

    def updateParameters(self):
        """
        M-step: Calculate a new parameters of the registration.
        """
        jacobianDamping = (self.dampingAnnealing) ** (self.iteration) * self.damping
        if (
            jacobianDamping < self.minDampingFactor
        ):  # lower limit of regularization to ensure stability of matrix inversion
            jacobianDamping = self.minDampingFactor

        self.X_desired = np.divide(self.PY, self.P1[:, None])
        # kinematic regularization (solve IK iteratively)
        dq = np.zeros(len(self.q))
        q = self.q
        ik_iterations = self.ik_iterations
        for i in range(0, ik_iterations):
            X_current = self.model.getPositions(q)
            X_error = self.X_desired - X_current
            jacobians = []
            for n in range(0, self.N):
                J_hat = self.model.getJacobian(q, n)
                jacobians.append(J_hat)
            J = np.vstack(jacobians)
            dq = dampedPseudoInverse(J, jacobianDamping) @ X_error.flatten()
            q = q + dq

            # @xwk: optionally incorporate stiffness
            # A = np.vstack(
            #     (
            #         J,
            #         np.sum(stiffnessMatrix, axis=0),
            #         np.tile(gravity, (1, len(X_current))) @ J,
            #     )
            # )
            # B = np.append(
            #     X_error.flatten(),
            #     (
            #         np.sum(stiffnessMatrix @ (self.q0 - q), axis=0),
            #         np.sum(X_current * gravity),
            #     ),
            # )

        # update generalized coordinates
        self.q = q

        # set the new targets
        self.computeTargets()

        # update objective function
        Lold = self.L
        self.L = (
            np.sum(np.log(self.Pden))
            + self.D * self.M * np.log(self.sigma2) / 2
            # - lambdaFactor / 2 * np.trace(np.transpose(self.W) @ self.G @ self.W)
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

    def getParameters(self):
        """
        Return the current estimate of the deformable transformation parameters.
        Returns
        -------
        self.T: numpy array of target points
        self.q: numpy array of corresponding generalized coordinates
        """
        return self.T, self.q
