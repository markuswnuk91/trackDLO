import os
import sys
import numpy as np
import numbers
from warnings import warn
import time
from sklearn.linear_model import Lasso, Ridge

try:
    sys.path.append(os.getcwd().replace("/src/tracking/krp", ""))
    from src.tracking.registration import NonRigidRegistration
    from src.utils.utils import initialize_sigma2
    from src.tracking.kpr.kinematicsModel import KinematicsModelDart
    from src.utils.utils import gaussian_kernel, initialize_sigma2
    from src.utils.utils import dampedPseudoInverse
    from src.localization.downsampling.mlle.mlle import Mlle
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
        model,
        q0=None,
        damping=None,
        stiffnessMatrix=None,
        gravity=None,
        groundLevel=None,
        constrainedNodeIndices=None,
        constrainedPositions=None,
        wCorrespondance=None,
        wStiffness=None,
        wGravity=None,
        wConstraint=None,
        minDampingFactor=None,
        dampingAnnealing=None,
        stiffnessAnnealing=None,
        gravitationalAnnealing=None,
        normalize=None,
        ik_iterations=None,
        log=None,
        method=None,  # "newton"(default), ik
        alpha=None,
        beta=None,
        gamma=None,
        knn=None,
        *args,
        **kwargs
    ):
        if type(qInit) is not np.ndarray or qInit.ndim > 1:
            raise ValueError("The degrees of freedom (q) must be a 1D numpy array.")
        if knn is not None and (not isinstance(knn, numbers.Number) or knn < 3):
            raise ValueError(
                "Expected at least 3 neighbors to reconstruct the local neighborhood instead got: {}".format(
                    knn
                )
            )
        self.qInit = qInit
        self.q = qInit.copy()
        self.dq = np.zeros(self.q.shape[0])
        self.q0 = qInit if q0 is None else q0
        self.groundLevel = np.array([0, 0, 0]) if groundLevel is None else groundLevel
        self.log = False if log is None else True
        self.method = method
        self.model = model
        X = model.getPositions(self.qInit)
        self.alpha = 1 if alpha is None else alpha

        super().__init__(X=X, *args, **kwargs)

        self.Dof = self.q.size

        self.damping = 1 if damping is None else damping
        self.ik_iterations = 3 if ik_iterations is None else ik_iterations
        self.stiffnessMatrix = (
            1 * np.eye(self.Dof) if stiffnessMatrix is None else stiffnessMatrix
        )
        self.gravity = np.array([0, 0, 0]) if gravity is None else gravity
        self.constraintNodeIndices = (
            None if constrainedNodeIndices is None else constrainedNodeIndices
        )
        if isinstance(constrainedPositions, (list, type(None))):
            self.constrainedPositions = (
                None if constrainedPositions is None else constrainedPositions
            )
        else:
            raise ValueError(
                "Expected to obtain a list of np.arrays for as costraied positions."
            )
        self.wCorrespondance = 1 if wCorrespondance is None else wCorrespondance
        self.wStiffness = 1 if wStiffness is None else wStiffness
        self.wGravity = 1 if wGravity is None else wGravity
        self.wConstraint = 1 if wConstraint is None else wConstraint

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

        # cpd_ik specific
        self.beta = 0.5 if beta is None else beta
        self.G = gaussian_kernel(X, self.beta)
        self.W = np.zeros((self.N, self.D))

        # spr_ik specific
        self.gamma = 0 if gamma is None else gamma
        self.knn = 7 if knn is None else knn
        self.Phi = Mlle(self.X, knn, 2).getAlignmentMatrix()

        if self.logging:
            self.log["q"] = [self.q]
            self.log["sigma2"] = [self.sigma2]

        self.initializeCorrespondances()

    def initializeCorrespondances(self):
        self.P = np.zeros((self.N, self.M))
        self.Pden = np.zeros((self.M))
        self.Pt1 = np.zeros((self.M,))
        self.P1 = np.zeros((self.N,))
        self.PY = np.zeros((self.N, self.D))
        self.Np = 0

    def reinitializeParameters(self):
        self.initializeCorrespondances()
        self.estimateCorrespondance()
        self.update_variance()

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
            T = self.model.getPositions(q)
            return T
        else:
            self.T = self.model.getPositions(self.q)
        return

    def updateParameters(self):
        """
        M-step: Calculate a new parameters of the registration.
        """
        jacobianDamping = (self.dampingAnnealing) ** (
            self.totalIterations
        ) * self.damping
        if (
            jacobianDamping < self.minDampingFactor
        ):  # lower limit of regularization to ensure stability of matrix inversion
            jacobianDamping = self.minDampingFactor
        wStiffness = (self.stiffnessAnnelealing) ** (
            self.totalIterations
        ) * self.wStiffness
        wGravity = (self.gravitationalAnnealing) ** (
            self.totalIterations
        ) * self.wGravity
        stiffnessMatrix = (self.stiffnessAnnelealing) ** (
            self.iteration
        ) * self.stiffnessMatrix

        dEGrav = np.zeros(self.Dof)

        # we can do this more efficnetly
        # for n in range(0, self.N):
        #     J = self.model.getJacobian(self.q, n)
        #     dEGrav += self.gravity @ J
        #     for m in range(0, self.M):
        #         A += self.P[n, m] * (J.T @ J)
        #         B += self.P[n, m] * (J.T @ (self.Y[m, :] - self.T[n, :]).T)

        # dq = np.zeros(self.Dof)
        # q = self.q
        # # dx = np.mean(self.PY / np.sum(self.P, axis=1)[:, np.newaxis], axis=0) - np.mean(
        # #     self.T, axis=0
        # # )
        # # dq[3:6] = dx
        # for n in range(0, self.ik_iterations):
        #     A = np.zeros((self.Dof, self.Dof))
        #     B = np.zeros(self.Dof)
        #     for n in range(0, self.N):
        #         Jn = self.model.getJacobian(q + dq, n)
        #         JnTJn = Jn.T @ Jn
        #         JnTJn_weighted = JnTJn * self.P1[n]
        #         A += JnTJn_weighted
        #         B += Jn.T @ (self.P[n, :] @ (self.Y - self.T[n, :])).T

        #     A += self.sigma2 * wStiffness * stiffnessMatrix
        #     B += (
        #         self.sigma2
        #         * wStiffness
        #         * stiffnessMatrix
        #         @ (np.zeros(self.Dof) - (q + dq))
        #     )
        #     AInvDamped = dampedPseudoInverse(A, jacobianDamping)
        #     # dq = dq - AInvDamped @ (
        #     #     (A + jacobianDamping**2 * np.eye(self.Dof)) @ dq - B
        #     # )
        #     dq = AInvDamped @ B  # newton iteration step q_t+1 = q_t - f(q_t)/f'(q_t)
        #     # dq[3:6] = dx
        # self.q = q + dq
        # # # update degrees of freedom
        # # self.updateDegreesOfFreedom()
        # # set the new targets
        # self.computeTargets()

        # # --------------------------------------
        if (self.method == "newton") or (self.method == None):
            for iter in range(0, self.ik_iterations):
                t_iter_start = time.time()
                # delta_x_desired = np.mean(
                #     self.PY / np.sum(self.P, axis=1)[:, np.newaxis], axis=0
                # ) - np.mean(self.T, axis=0)
                # dq_trans = delta_x_desired
                # self.q[3:6] = self.q[3:6] + dq_trans
                # self.computeTargets()
                A = np.zeros((self.Dof, self.Dof))
                B = np.zeros(self.Dof)
                Jn_list = []
                JnTJn_list = []
                for n in range(0, self.N):
                    Jn = self.model.getJacobian(self.q, n)
                    JnTJn = Jn.T @ Jn
                    JnTJn_weighted = JnTJn * self.P1[n]
                    A += JnTJn_weighted
                    Jn_list.append(self.model.getJacobian(self.q, n))
                    JnTJn_list.append(JnTJn)
                    B += Jn.T @ (self.P[n, :] @ (self.Y - self.T[n, :])).T
                    B += (
                        self.sigma2
                        * wGravity
                        * Jn.T
                        @ (
                            (
                                self.gravity
                                @ np.diag(np.sign((self.groundLevel - self.T[n, :])))
                            )
                        )
                    )

                A += self.sigma2 * wStiffness * stiffnessMatrix
                B += self.sigma2 * wStiffness * stiffnessMatrix @ (self.q0 - self.q)

                # position constraints
                if self.constraintNodeIndices is not None:
                    for i, constraintNodeIndex in enumerate(self.constraintNodeIndices):
                        Jc = self.model.getJacobian(self.q, constraintNodeIndex)
                        A += self.wConstraint * Jc.T @ Jc
                        B += (
                            self.wConstraint
                            * Jc.T
                            @ (
                                self.constrainedPositions[i]
                                - self.T[constraintNodeIndex, :]
                            )
                        )
                # B += (
                #     self.sigma2
                #     * wStiffness
                #     * stiffnessMatrix
                #     @ (np.zeros(len(self.q)) - self.q)
                # )

                # t_matrix_assembly_end = time.time()
                # print(
                #     "Time for matrix assembly: {}".format(
                #         t_matrix_assembly_end - t_matrix_assembly_start
                #     )
                # )
                # t_pinv_start = time.time()
                AInvDamped = dampedPseudoInverse(A, jacobianDamping)
                # t_pinv_end = time.time()
                # print("Time for matrix inversion: {}".format(t_pinv_end - t_pinv_start))
                # ridge = Lasso(alpha=0.1)
                # ridge.fit(A, B)
                # self.dq = ridge.coef_
                self.dq = AInvDamped @ B
                self.dq[3:6] = np.mean(
                    (self.PY - np.diag(self.P1) @ self.T), axis=0
                ) / np.sum(self.P)
                # update degrees of freedom
                self.updateDegreesOfFreedom()
                # set the new targets
                self.computeTargets()
                t_iter_end = time.time()
                print("Time per iteration: {}".format(t_iter_end - t_iter_start))
            self.update_variance()
        if self.method == "ik":
            for iter in range(0, self.ik_iterations):
                # t_iter_start = time.time()
                # self.dq = np.zeros(self.Dof)
                # dP1 = np.diag(self.P1)
                # self.dq[3:6] = np.mean(np.linalg.inv(dP1) @ self.PY) - np.mean(self.T)
                # self.updateDegreesOfFreedom()
                # self.computeTargets()
                # determine error
                # dP1 = np.diag(self.P1)

                # error = np.linalg.inv(dP1) @ self.PY - self.T
                A = np.diag(self.P1) + self.alpha * self.sigma2 * np.eye(self.N)
                B = self.PY - np.dot(np.diag(self.P1), self.T)
                error = np.linalg.solve(A, B)
                X_target = self.T + error
                # assembly Jacobian
                Jn_list = []
                for n in range(0, self.N):
                    Jn = self.model.getJacobian(self.q, n)
                    Jn_list.append(Jn)
                J = np.vstack(Jn_list)
                pInv = dampedPseudoInverse(J, jacobianDamping)
                self.dq = (
                    pInv @ error.flatten()
                    + self.sigma2 * wStiffness * stiffnessMatrix @ (self.q0 - self.q)
                )
                # update degrees of freedom
                self.updateDegreesOfFreedom()
                # set the new targets
                self.computeTargets()
                # t_iter_end = time.time()
                # print("Time per iteration: {}".format(t_iter_end - t_iter_start))
            self.update_variance()
            print("Sigma: {}".format(self.sigma2))
        if self.method == "gradient_descend":
            for iter in range(0, self.ik_iterations):
                t_iter_start = time.time()
                # assembly Jacobian
                Jn_list = []
                for n in range(0, self.N):
                    Jn = self.model.getJacobian(self.q, n)
                    Jn_list.append(Jn)
                J = np.vstack(Jn_list)
                # gradient = np.zeros(self.Dof)
                # for k in range(0, self.Dof):
                #     for m in range(0, self.M):
                #         for n in range(0, self.N):
                #             gradient[k] += (
                #                 self.P[n, m]
                #                 * J[3 * n : 3 * (n + 1), k].T
                #                 @ (self.Y[m, :] - self.T[n, :])

                gradient = J.T @ (self.PY - np.dot(np.diag(self.P1), self.T)).flatten()
                learning_rate = 0.001
                self.dq = learning_rate * gradient
                # self.dq[0:3] = np.zeros(3)
                # self.dq[3:6] = np.mean(
                #     np.linalg.inv(np.diag(self.P1)) @ self.PY
                # ) - np.mean(self.T)
                # update degrees of freedom
                self.updateDegreesOfFreedom()
                # set the new targets
                self.computeTargets()
                t_iter_end = time.time()
                print("Time per iteration: {}".format(t_iter_end - t_iter_start))
                self.update_variance()
            print("Sigma: {}".format(self.sigma2))
        if self.method == "cpd_ik":
            for iter in range(0, self.ik_iterations):
                t_iter_start = time.time()
                # error = np.linalg.inv(dP1) @ self.PY - self.T
                A = np.dot(
                    np.diag(self.P1), self.G
                ) + self.alpha * self.sigma2 * np.eye(self.N)
                B = self.PY - np.dot(np.diag(self.P1), self.T)
                self.W = np.linalg.solve(A, B)
                error = np.dot(self.G, self.W)
                X_target = self.T + error
                # assembly Jacobian
                Jn_list = []
                for n in range(0, self.N):
                    Jn = self.model.getJacobian(self.q, n)
                    Jn_list.append(Jn)
                J = np.vstack(Jn_list)
                pInv = dampedPseudoInverse(J, jacobianDamping)
                self.dq = (
                    pInv @ error.flatten()
                    + self.sigma2 * wStiffness * stiffnessMatrix @ (self.q0 - self.q)
                )
                # update degrees of freedom
                self.updateDegreesOfFreedom()
                # set the new targets
                self.computeTargets()
                t_iter_end = time.time()
                print("Time per iteration: {}".format(t_iter_end - t_iter_start))
            self.update_variance()
            print("Sigma: {}".format(self.sigma2))
        if self.method == "spr_ik":
            # t_iter_start = time.time()
            # error = np.linalg.inv(dP1) @ self.PY - self.T
            A = (
                np.dot(np.diag(self.P1), self.G)
                + self.alpha * self.sigma2 * np.eye(self.N)
                + self.gamma * self.sigma2 * np.dot(self.Phi, self.G)
            )
            B = (
                self.PY
                - np.dot(np.diag(self.P1), self.T)
                - self.gamma * self.sigma2 * np.dot(self.Phi, self.T)
            )
            self.W = np.linalg.solve(A, B)
            error = np.dot(self.G, self.W)
            X_target = self.T + error
            q_0 = self.q
            # P1_norm = (self.P1 - np.min(self.P1)) / (
            #     np.max(self.P1) - np.min(self.P1)
            # )
            # Wp = np.diag(np.repeat(P1_norm, self.D))
            # Wp = np.diag(np.repeat(np.exp(-1 * (1 - P1_norm)), self.D))
            P1_norm = self.P1 / np.mean(self.P1)
            # P1_norm[3:6] = np.ones(3)
            Wp = np.diag(np.repeat(1 / (1 + (np.exp(-(P1_norm)))), self.D))
            for iter in range(0, self.ik_iterations):
                error = X_target - self.T
                # assembly Jacobian
                Jn_list = []
                for n in range(0, self.N):
                    Jn = self.model.getJacobian(self.q, n)
                    Jn_list.append(Jn)
                J = np.vstack(Jn_list)
                # pInv = dampedPseudoInverse(J, jacobianDamping)
                # self.dq = (
                #     pInv @ error.flatten()
                #     + self.sigma2 * wStiffness * stiffnessMatrix @ (self.q0 - self.q)
                # )
                # A = Wp @ J
                # # A += wStiffness * stiffnessMatrix @ (self.q0 - self.q)
                # pInv = dampedPseudoInverse(A, jacobianDamping)
                # self.dq = (
                #     pInv @ Wp @ error.flatten()
                #     + self.sigma2 * wStiffness * stiffnessMatrix @ (self.q0 - self.q)
                # )
                # --------------------------------------------------------------
                A = (
                    J.T @ Wp.T @ Wp @ J
                    + wStiffness * stiffnessMatrix
                    + jacobianDamping**2 * np.eye(self.Dof)
                )
                # A += wStiffness * stiffnessMatrix @ (self.q0 - self.q)
                # pInv = dampedPseudoInverse(A, jacobianDamping)
                dq_stiff = q_0 - self.q
                dq_stiff[0:6] = np.zeros(6)
                self.dq = np.linalg.inv(A) @ (
                    J.T @ Wp.T @ Wp @ error.flatten()
                    + wStiffness * stiffnessMatrix @ (dq_stiff)
                )
                # --------------------------------------------------------------
                # self.dq[0:3] = self.dq[9:12]
                # self.dq[6:9] = self.dq[9:12]
                # update degrees of freedom
                self.updateDegreesOfFreedom()
                # set the new targets
                self.computeTargets()
                # t_iter_end = time.time()
                # print("Time per iteration: {}".format(t_iter_end - t_iter_start))
            self.update_variance()
            # print("Sigma: {}".format(self.sigma2))
        # -------------------------------------------------------------------
        # for iteration in range(0, 3):
        #     dP1 = np.diag(self.P1)
        #     dP1Inv = np.linalg.inv(dP1)
        #     A = np.zeros((self.D, self.Dof))
        #     B = np.zeros((self.N, self.D))
        #     J = []
        #     for n in range(0, self.N):
        #         Jn = self.model.getJacobian(self.q, n)
        #         B[n, :] = (self.P[n, :] @ (self.Y - self.T[n, :])) / np.sum(
        #             self.P[n, :]
        #         )
        #         J.append(Jn)
        #     J = np.vstack(J)
        #     #     # )
        #     # Jn = self.model.getJacobian(self.q, n)
        #     # Jn_weighted = Jn * self.P1[n]
        #     # A += Jn_weighted
        #     # B += self.P[n, :] @ (self.Y - self.T[n, :])

        #     # A += self.sigma2 * wStiffness * stiffnessMatrix
        #     # B += self.sigma2 * wStiffness * stiffnessMatrix @ (self.qInit - self.q)
        #     self.dq = dampedPseudoInverse(J, 0.1)
        #     self.dq[3:6] = np.mean(dP1Inv @ self.PY, axis=0) - np.mean(self.T, axis=0)
        #     self.updateDegreesOfFreedom()
        #     self.computeTargets()
        #     self.update_variance()
        # ----------------------------------
        # dP1 = np.diag(self.P1)
        # dP1Inv = np.linalg.inv(dP1)
        # X_d = self.T + dP1Inv @ (self.PY - np.dot(dP1, self.T))
        # for nit in range(0, self.ik_iterations):
        #     J = []
        #     for n in range(0, self.N):
        #         J_n = self.model.getJacobian(self.q, n)
        #         J.append(J_n)
        #     J = np.vstack(J)
        #     # self.dq[:6] = dampedPseudoInverse(J[:, :6], 0.7) @ (
        #     #     X_d.flatten() - self.T.flatten()
        #     # )
        #     self.dq = dampedPseudoInverse(J, 0.1) @ (
        #         X_d - self.T
        #     ).flatten() + self.sigma2 * wStiffness * stiffnessMatrix @ (
        #         np.zeros(self.Dof) - (self.q)
        #     )
        #     self.dq[3:6] = np.mean(dP1Inv @ self.PY, axis=0) - np.mean(self.T, axis=0)
        #     self.updateDegreesOfFreedom()
        #     self.computeTargets()
        # self.update_variance()

        if self.logging:
            self.log["q"].append(self.q)
            self.log["sigma2"].append(self.sigma2)
            self.log["T"].append(self.T)

    def update_variance(self):
        """
        Update the variance of the mixture model using the new estimate of the deformable transformation.
        See the update rule for sigma2 in Eq. 23 of of https://arxiv.org/pdf/0905.2635.pdf.
        """

        # The original CPD paper does not explicitly calculate the objective functional.
        # This functional includes terms from both the negative log-likelihood and
        # the Gaussian kernel used for regularization.
        Lold = self.L
        self.L = np.sum(-np.log(self.Pden)) - self.D * self.M * np.log(self.sigma2) / 2
        self.diff = np.abs((self.L - Lold) / self.L)

        # Optionally we could use the difference between the current and previous
        # estimate of the variance as a proxy to test for convergence.
        # qprev = self.sigma2 #comment in to use variance to test convergence

        yPy = np.dot(
            np.transpose(self.Pt1), np.sum(np.multiply(self.Y, self.Y), axis=1)
        )
        xPx = np.dot(np.transpose(self.P1), np.sum(np.multiply(self.T, self.T), axis=1))
        trPXY = np.sum(np.multiply(self.T, self.PY))

        self.sigma2 = (xPx - 2 * trPXY + yPy) / (self.Np * self.D)

        if self.sigma2 <= 0:
            self.sigma2 = self.tolerance / 10

        # self.diff = np.abs(self.sigma2 - qprev) # comment in to use variance to test convergence

    def getParameters(self):
        """
        Return the current estimate of the deformable transformation parameters.
        Returns
        -------
        self.T: numpy array of target points
        self.q: numpy array of corresponding generalized coordinates
        """
        return self.T, self.q

    def getResults(self):
        result = {
            "X": self.X,
            "T": self.T,
            "q": self.q,
            "sigma2": self.sigma2,
            "runtimes": self.runTimes,
        }
        if self.logging:
            result["log"] = self.log
        return result
