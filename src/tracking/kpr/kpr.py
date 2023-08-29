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
        damping=None,
        stiffnessMatrix=None,
        gravity=None,
        wCorrespondance=None,
        wStiffness=None,
        wGravity=None,
        minDampingFactor=None,
        dampingAnnealing=None,
        stiffnessAnnealing=None,
        gravitationalAnnealing=None,
        normalize=None,
        ik_iterations=None,
        *args,
        **kwargs
    ):
        if type(qInit) is not np.ndarray or qInit.ndim > 1:
            raise ValueError("The degrees of freedom (q) must be a 1D numpy array.")

        self.qInit = qInit
        self.q = qInit.copy()
        self.dq = np.zeros(self.q.shape[0])
        self.deltaq = np.delete(qInit.copy(), [3, 4, 5])
        self.model = model
        X = model.getPositions(self.qInit)
        super().__init__(X=X, *args, **kwargs)

        self.Dof = self.q.size

        self.damping = 1 if damping is None else damping
        self.ik_iterations = 3 if ik_iterations is None else ik_iterations
        self.stiffnessMatrix = (
            1 * np.eye(self.Dof) if stiffnessMatrix is None else stiffnessMatrix
        )
        self.gravity = np.array([0, 0, 0]) if gravity is None else gravity

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

        for n in range(0, self.ik_iterations):
            delta_x_desired = np.mean(
                self.PY / np.sum(self.P, axis=1)[:, np.newaxis], axis=0
            ) - np.mean(self.T, axis=0)
            dq_trans = delta_x_desired
            self.q[3:6] = self.q[3:6] + dq_trans
            self.computeTargets()
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

            A += self.sigma2 * wStiffness * stiffnessMatrix
            B += self.sigma2 * wStiffness * stiffnessMatrix @ (self.qInit - self.q)

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
            self.dq[3:6] = np.zeros(3)
            # update degrees of freedom
            self.updateDegreesOfFreedom()
            # set the new targets
            self.computeTargets()

        self.update_variance()

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
