import os
import sys
import numpy as np
import numbers
from warnings import warn

try:
    sys.path.append(os.getcwd().replace("/src/tracking/cpd", ""))
    from src.utils.utils import initialize_sigma2
    from src.tracking.kpr.kinematicsModel import KinematicsModelDart
    from src.utils.utils import gaussian_kernel, initialize_sigma2
except:
    print("Imports for KPR failed.")
    raise


class KinematicsPreservingRegistration(object):
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

    sigma2: float (positive)
        Variance of the Gaussian mixture model

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
        q0,
        Y,
        model,
        max_iterations=None,
        tolerance=None,
        damping=None,
        stiffnessMatrix=None,
        gravity=None,
        sigma2=None,
        mu=None,
        wCorrespondance=None,
        wStiffness=None,
        wGravity=None,
        minDampingFactor=None,
        dampingAnnealing=None,
        stiffnessAnnealing=None,
        gravitationalAnnealing=None,
        normalize=False,
        ik_iterations=None,
        *args,
        **kwargs
    ):
        if type(qInit) is not np.ndarray or qInit.ndim > 1:
            raise ValueError("The degrees of freedom (q) must be a 1D numpy array.")

        if type(Y) is not np.ndarray or Y.ndim != 2:
            raise ValueError("The target point cloud (Y) must be a 2D numpy array.")
        if Y.shape[0] < Y.shape[1]:
            raise ValueError(
                "The dimensionality is larger than the number of points. Possibly the wrong orientation of Y."
            )
        if Y.shape[0] < Y.shape[1]:
            raise ValueError(
                "The dimensionality is larger than the number of points. Possibly the wrong orientation of Y."
            )
        if max_iterations is not None and (
            not isinstance(max_iterations, numbers.Number) or max_iterations < 0
        ):
            raise ValueError(
                "Expected a positive integer for max_iterations instead got: {}".format(
                    max_iterations
                )
            )
        elif isinstance(max_iterations, numbers.Number) and not isinstance(
            max_iterations, int
        ):
            warn(
                "Received a non-integer value for max_iterations: {}. Casting to integer.".format(
                    max_iterations
                )
            )
            max_iterations = int(max_iterations)

        if tolerance is not None and (
            not isinstance(tolerance, numbers.Number) or tolerance < 0
        ):
            raise ValueError(
                "Expected a positive float for tolerance instead got: {}".format(
                    tolerance
                )
            )

        if sigma2 is not None and (
            not isinstance(sigma2, numbers.Number) or sigma2 <= 0
        ):
            raise ValueError(
                "Expected a positive value for sigma2 instead got: {}".format(sigma2)
            )

        if mu is not None and (not isinstance(mu, numbers.Number) or mu < 0 or mu >= 1):
            raise ValueError(
                "Expected a value between 0 (inclusive) and 1 (exclusive) for mu instead got: {}".format(
                    mu
                )
            )

        self.qInit = qInit
        self.q0 = qInit.copy() if q0 is None else q0
        self.q = qInit.copy()
        self.dq = np.zeros(self.q.shape[0])
        self.deltaq = np.delete(qInit.copy(), [3, 4, 5])
        self.model = model
        self.Y = Y
        self.X = model.getPositions(self.qInit)
        self.T = model.getPositions(self.q)
        if normalize == True:
            self.YMean = np.mean(self.Y)
            self.Y = self.Y - self.YMean
            self.XMean = np.mean(self.X)
            self.X = self.X - self.XMean
            self.TMean = np.mean(self.T)
            self.T = self.T - self.TMean
        self.Dof = self.q.size
        (self.N, _) = self.T.shape
        (self.M, self.D) = self.Y.shape
        self.tolerance = 10e-5 if tolerance is None else tolerance
        self.max_iterations = 100 if max_iterations is None else max_iterations
        self.iteration = 0
        self.damping = 1 if damping is None else damping
        self.stiffnessMatrix = (
            0 * np.eye(self.Dof) if stiffnessMatrix is None else stiffnessMatrix
        )
        self.gravity = np.array([0, 0, 0]) if gravity is None else gravity
        self.sigma2 = initialize_sigma2(self.T, self.Y) if sigma2 is None else sigma2
        self.mu = 0.0 if mu is None else mu
        self.wCorrespondance = 1 if wCorrespondance is None else wCorrespondance
        self.wStiffness = 1 if wStiffness is None else wStiffness
        self.wGravity = 1 if wGravity is None else wGravity
        self.minDampingFactor = 1 if minDampingFactor is None else minDampingFactor
        self.dampingAnnealing = 0.97 if dampingAnnealing is None else dampingAnnealing
        self.stiffnessAnnelealing = (
            0.97 if stiffnessAnnealing is None else stiffnessAnnealing
        )
        self.gravitationalAnnealing = (
            0.97 if gravitationalAnnealing is None else gravitationalAnnealing
        )
        self.diff = np.inf
        self.L = -np.inf

        self.P = np.zeros((self.N, self.M))
        self.Pden = np.zeros((self.M))
        self.Pt1 = np.zeros((self.M,))
        self.P1 = np.zeros((self.N,))
        self.Np = 0
        self.PY = np.zeros((self.N, self.D))
        self.W = np.zeros((self.Dof, self.D))
        self.Gq = np.eye(self.Dof)

        self.WRigid = np.zeros((self.N, self.D))
        self.WNonrigid = np.zeros((self.N, self.D))
        self.q_t = np.zeros(self.Dof - 3)
        self.q_t[:3] = self.qInit[:3]
        self.q_t[3:] = self.qInit[6:]
        self.deltaX_t = np.zeros((self.N, self.D))
        self.deltaX_Rigid_t = np.zeros(self.D)
        self.deltaX_Nonrigid_t = np.zeros((self.N, self.D))
        self.ik_iterations = 1 if ik_iterations is None else ik_iterations

    def register(self, callback=None):
        """
        Peform the registration

        Attributes
        ----------
        callback: function
            A function that will be called after each iteration.
            Can be used to visualize the registration process.

        Returns
        -------
        self.T: numpy array
            MxD array of transformed source points.

        registration_parameters:
            Returned params dependent on registration method used.
        """
        self.computeTargets()
        while self.iteration < self.max_iterations and not self.isConverged():
            self.iterate()
            if callable(callback):
                callback()

        return self.T, self.getParameters()

    def iterate(self):
        """
        Perform one iteration of the registration.
        """
        self.estimateCorrespondance()
        self.updateParameters()
        self.computeTargets()
        self.iteration += 1

    def isConverged(self):
        """
        Checks if change of cost function is below the defined tolerance
        """
        return self.diff < self.tolerance

    def estimateCorrespondance(self, normalize=True):
        """
        E-step: Compute the expectation step  of the EM algorithm.
        """
        if normalize:
            # normalize to 0 mean
            Y_hat = self.Y - np.mean(self.Y)
            T_hat = self.T - np.mean(self.T)
            # normalize to 0 variance
            scalingFactor_T = np.sqrt(np.sum(self.T**2) / self.N)
            scalingFactor_Y = np.sqrt(np.sum(self.Y**2) / self.M)
            Y_hat = Y_hat / scalingFactor_Y
            T_hat = T_hat / scalingFactor_T
            P = np.sum((Y_hat[None, :, :] - T_hat[:, None, :]) ** 2, axis=2)

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
            P = np.sum((self.Y[None, :, :] - self.T[:, None, :]) ** 2, axis=2)

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
        # qNonrigid = np.zeros(self.Dof - 3)
        # qNonrigid[:3] = self.qInit[:3] + self.deltaq[:3]
        # qNonrigid[3:] = self.qInit[6:] + self.deltaq[3:]
        # q = np.zeros(self.Dof)
        # q[:3] = qNonrigid[:3]
        # q[3:6] = self.X[0, :] + self.WRigid[0, :]
        # q[6:] = qNonrigid[3:]
        # for n in range(0, self.N):
        #     self.T[n, :] = self.model.getPositions(q)[n, :]
        return

    def dampedPseudoInverse(self, J, dampingFactor):
        dim = J.shape[0]
        dampedPseudoInverse = J.T @ np.linalg.inv(
            J @ J.T + dampingFactor**2 * np.eye(dim)
        )
        return dampedPseudoInverse

    def updateParameters(self, method="gmm"):
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
        stiffnessMatrix = (self.stiffnessAnnelealing) ** (
            self.iteration
        ) * self.stiffnessMatrix

        if method == "together":
            A = np.zeros((self.Dof, self.Dof))
            B = np.zeros(self.Dof)
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
            A += wStiffness * stiffnessMatrix
            B += (
                wStiffness * stiffnessMatrix @ (self.q0 - self.q)
            )  # add stiffness term for right side
            B += wGravity * dEGrav  # add gravitational term
            AInvDamped = self.dampedPseudoInverse(A, jacobianDamping)
            self.dq = AInvDamped @ B
            # update degrees of freedom
            self.updateDegreesOfFreedom()
            # set the new targets
            self.computeTargets()
        elif method == "priorFirst":
            dqRot = np.concatenate((self.dq[:3], self.dq[6:]))  # only rotational dofs
            x0Num = 0
            A = np.zeros((self.Dof - 3, self.Dof - 3))
            B = np.zeros(self.Dof - 3)
            dEGrav = np.zeros(self.Dof - 3)
            J = []
            q0Rot = np.concatenate((self.q0[:3], self.q0[6:]))
            stiffnessMatrix = self.stiffnessMatrix
            stiffnessMatrix = np.delete(stiffnessMatrix, [3, 4, 5], axis=0)
            stiffnessMatrix = np.delete(stiffnessMatrix, [3, 4, 5], axis=1)

            # translation
            for n in range(0, self.N):
                J_hat = self.model.getJacobian(self.q, n)
                J.append(
                    np.hstack((J_hat[:, :3], J_hat[:, 6:]))
                )  # only for rotational dofs
            for m in range(0, self.M):
                x0Num += self.P[0, m] * (self.Y[m, :] - self.T[0, :] - J[0] @ dqRot)
            dx0 = 0.1 * x0Num / (np.sum(self.P[0, :]))
            self.q[3:6] = dx0
            self.computeTargets()
            self.estimateCorrespondance()
            # rotation
            for n in range(0, self.N):
                dEGrav += self.gravity @ J[n]
                for m in range(0, self.M):
                    A += self.wCorrespondance * self.P[n, m] * (J[n].T @ J[n])
                    B += (
                        self.wCorrespondance
                        * self.P[n, m]
                        * ((J[n].T @ (self.Y[m, :] - self.T[n, :] - dx0).T))
                    )
            AInvDamped = self.dampedPseudoInverse(A, jacobianDamping)
            A += wStiffness * stiffnessMatrix
            B += (
                wStiffness * stiffnessMatrix @ (q0Rot - dqRot)
            )  # add stiffness term for right side
            B += wGravity * dEGrav  # add gravitational term
            dqRot = AInvDamped @ B

            self.dq[:3] = dqRot[:3]
            self.dq[3:6] = dx0
            self.dq[6:] = dqRot[3:]
            self.updateDegreesOfFreedom()
            # set the new targets
            self.computeTargets()
        elif method == "separated":
            dqRot = np.concatenate((self.dq[:3], self.dq[6:]))  # only rotational dofs
            x0Num = 0
            A = np.zeros((self.Dof - 3, self.Dof - 3))
            B = np.zeros(self.Dof - 3)
            dEGrav = np.zeros(self.Dof - 3)
            J = []
            q0Rot = np.concatenate((self.q0[:3], self.q0[6:]))
            stiffnessMatrix = self.stiffnessMatrix
            stiffnessMatrix = np.delete(stiffnessMatrix, [3, 4, 5], axis=0)
            stiffnessMatrix = np.delete(stiffnessMatrix, [3, 4, 5], axis=1)

            # translation
            for n in range(0, self.N):
                J_hat = self.model.getJacobian(self.q, n)
                J.append(
                    np.hstack((J_hat[:, :3], J_hat[:, 6:]))
                )  # only for rotational dofs
                for m in range(0, self.M):
                    x0Num += self.P[n, m] * (
                        self.Y[m, :] - self.X[n, :] - self.WNonrigid[n]
                    )
            dx0 = x0Num / (np.sum(self.P))
            self.WRigid += 0.9 * (np.tile(dx0, (self.N, 1)) - self.WRigid)
            self.computeTargets()
            self.estimateCorrespondance()
            # rotation
            for n in range(0, self.N):
                dEGrav += self.gravity @ J[n]
                for m in range(0, self.M):
                    A += self.wCorrespondance * self.P[n, m] * (J[n].T @ J[n])
                    B += (
                        self.wCorrespondance
                        * self.P[n, m]
                        * ((J[n].T @ (self.Y[m, :] - self.T[n, :]).T))
                    )
            AInvDamped = self.dampedPseudoInverse(A, jacobianDamping)
            A += wStiffness * stiffnessMatrix
            B += (
                wStiffness * stiffnessMatrix @ (q0Rot - dqRot)
            )  # add stiffness term for right side
            B += wGravity * dEGrav  # add gravitational term
            dqRot = AInvDamped @ B

            # # regularize update velocity
            # W = np.zeros((self.N, self.D))
            # G = gaussian_kernel(self.X, 2)
            # dX = np.zeros((self.N, self.D))
            # for n in range(0, self.N):
            #     J_hat = self.model.getJacobian(self.q, n)
            #     J = np.hstack((J_hat[:, :3], J_hat[:, 6:]))  # only for rotational dofs
            #     dX[n, :] = dx0 + J @ dqRot
            # W = np.linalg.inv(G) @ dX
            # dx0 = (G @ W)[0, :]
            # convert to skeleton DOFs
            # self.dq[:3] = dqRot[:3]
            self.dq[:3] = dqRot[:3]
            self.dq[3:6] = dx0
            self.dq[6:] = dqRot[3:]

            self.deltaq += dqRot
            self.q[:3] = self.qInit[:3] + self.deltaq[:3]
            self.q[6:] = self.qInit[6:] + self.deltaq[3:]
            WNonrigid = []
            for Jn in J:
                xdot_n = Jn @ dqRot
                WNonrigid.append(xdot_n)
            self.WNonrigid += np.vstack(WNonrigid)
            # set the new targets
            self.computeTargets()

        if method == "absolute":
            A = np.zeros((self.Dof - 3, self.Dof - 3))
            B = np.zeros(self.Dof - 3)
            J = []
            for n in range(0, self.N):
                J_hat = self.model.getJacobian(self.q, n)
                J.append(
                    np.hstack((J_hat[:, :3], J_hat[:, 6:]))
                )  # only for rotational dofs

            # solve translation
            deltaX_Rigid_t_Nominator = 0
            for n in range(0, self.N):
                for m in range(0, self.M):
                    deltaX_Rigid_t_Nominator += self.P[n, m] * (
                        self.Y[m, :] - self.X[n, :] - self.deltaX_Nonrigid_t[n, :]
                    )
            self.deltaX_Rigid_t = deltaX_Rigid_t_Nominator / (np.sum(self.P))
            # update T
            for n in range(0, self.N):
                self.T[n, :] = (
                    self.X[n, :] + self.deltaX_Rigid_t + self.deltaX_Nonrigid_t[n, :]
                )
            # update correspondances
            self.estimateCorrespondance()
            # update q
            self.q[3:6] = self.T[0, :]
            J = []
            for n in range(0, self.N):
                J_hat = self.model.getJacobian(self.q, n)
                J.append(
                    np.hstack((J_hat[:, :3], J_hat[:, 6:]))
                )  # only for rotational dofs

            # solve for rotations
            for n in range(0, self.N):
                for m in range(0, self.M):
                    A += self.P[n, m] * (J[n].T @ J[n])
                    B += self.P[n, m] * J[n].T @ (self.Y[m, :] - self.T[n, :]).T

            self.dq = self.dampedPseudoInverse(A, jacobianDamping) @ B
            qNew = np.zeros(self.Dof)
            qNew[:3] = self.q_t[:3] + self.dq[:3]
            qNew[3:6] = self.X[0, :] + self.deltaX_Rigid_t
            qNew[6:] = self.q_t[3:] + self.dq[3:]
            qOld = np.zeros(self.Dof)
            qOld[:3] = self.q_t[:3]
            qOld[3:6] = self.X[0, :] + self.deltaX_Rigid_t
            qOld[6:] = self.q_t[3:]

            dX_Nonrigid = (
                self.model.getPositions(qNew)[n, :]
                - self.model.getPositions(qOld)[n, :]
            )
            self.deltaX_Nonrigid_t = self.deltaX_Nonrigid_t + dX_Nonrigid
            # update T
            for n in range(0, self.N):
                self.T[n, :] = (
                    self.X[n, :] + self.deltaX_Rigid_t + self.deltaX_Nonrigid_t[n, :]
                )

            self.q_t = self.q_t + self.dq  # update q

            self.q[:3] = self.q_t[:3]
            self.q[3:6] = self.T[0, :]
            self.q[6:] = self.q_t[3:]

        if method == "iterative":
            q = self.q
            ik_iterations = 5
            for i in range(ik_iterations):
                A = np.zeros((self.Dof, self.Dof))
                B = np.zeros(self.Dof)
                dEGrav = np.zeros(self.Dof)
                for n in range(0, self.N):
                    J = self.model.getJacobian(q, n)
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
                A += wStiffness * stiffnessMatrix
                B += (
                    wStiffness * stiffnessMatrix @ (self.q0 - self.q)
                )  # add stiffness term for right side
                B += wGravity * dEGrav  # add gravitational term
                AInvDamped = self.dampedPseudoInverse(A, jacobianDamping)
                self.dq = AInvDamped @ B
                q += self.dq
                self.T = self.computeTargets(q)
            self.q = q
            # set the new targets
            self.computeTargets()
        if method == "gmm":
            # dP1 = np.diag(self.P1)
            # A = dP1
            # B = self.PY - np.dot(dP1, self.X)
            # self.X_desired = np.linalg.solve(A, B)
            self.X_desired = np.divide(self.PY, self.P1[:, None])
            # kinematic regularization
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
