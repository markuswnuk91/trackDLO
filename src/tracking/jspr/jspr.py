import os
import sys
import numpy as np
import numbers
from warnings import warn
import dartpy as dart

try:
    sys.path.append(os.getcwd().replace("/src/tracking/cpd", ""))
    from src.tracking.utils.utils import gaussian_kernel, initialize_sigma2
except:
    print("Imports for JSPR failed.")
    raise


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
        Placeholder for child classes.
        """
        self.skel.setPositions(q)
        J = np.zeros((3, self.Dof))
        # if n == 0:
        #     dartJacobian = (
        #         np.linalg.inv(self.skel.getJoint(0).getRelativeTransform().rotation())
        #         @ self.skel.getBodyNode(n).getWorldJacobian(np.array([0, 0, 0]))[3:6, :]
        #     )
        # else:
        #     dartJacobian = self.skel.getBodyNode(n).getWorldJacobian(
        #         np.array([0, 0, 0])
        #     )[3:6, :]

        # dartJacobian = (
        #     np.linalg.inv(self.skel.getJoint(n).getRelativeTransform().rotation())
        #     @ self.skel.getBodyNode(n).getWorldJacobian(np.array([0, 0, 0]))[3:6, :]
        # )

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


class JacobianBasedStructurePreservingRegistration(object):
    """Base class for non-rigid registration for DLO
    Implementation of the Jacobian-Based Structure Preserved Registration (JSPR) algorithm.

    Attributes:
    -------------
    q: numpy array
        Nx1 array of degrees of freedom

    q_dot: numpy array
        Nx1 array of velocities of degrees of freedom

    Y: numpy array
        MxD array of data points (e.g. 3D point cloud)

    T: numpy array
        NxD array of target points (transformed source points)

    N: int
        Number of degrees of freedom

    M: int
        Number of data points

    D: int
        Dimensionality of source and target points

    model: kinematic model class
        a kinematic model that provides functionality to calculate positions X and Jacobians J from gernalized coordinates q.
        The provided model must provide the following two functions
            1) self.getPositions(q): returns a NxD np.array of spatial positions X from a (DoFyx1) np.array of generalized coordinates q
            2) self.getJacobian(q,n): returns the jacobian towards the spatatial position X(n,:) from a (DoFyx1) np.array of generalized coordinates q and position information n

    iterations: int
        The current iteration throughout the registration

    max_iterations: int
        Maximum number of iterations the registration performs before terminating

    tolerance: float (positive)
        tolerance for checking convergence.
        Registration will terminate once the difference between
        consecutive objective function values falls within this tolerance.

    vis: bool
        visualization for every iteration

        alpha: float (positive)
        Represents the trade-off between the goodness of maximum likelihood fit and regularization.

    lambdaFactor: float (positive)
        Regularization factor for the Global Regularization (CPD).
        A higher lambdaFactor enforces stronger global coupling (coherent movement)between points.

    beta: float(positive)
        Width of the Gaussian kernel for global regularization.
        A higher beta enforces a stronger coherence in the movement of the points, the points behave "more stiff".

    sigma2: float (positive)
        Variance of the Gaussian mixture model

    diff: float (positive)
        The absolute normalized difference between the current and previous objective function values.

    L: float
        The log-likelyhood of the dataset probability given the parameterization. SPR aims to update the parameters such that they maximize this value.

    P: numpy array
        MxN array of probabilities.
        P[m, n] represents the probability that the m-th source point
        corresponds to the n-th target point.

    mu: float (between 0 and 1)
        Contribution of the uniform distribution to account for outliers.
        Valid values span 0 (inclusive) and 1 (exclusive).

    lambdaAnnealing: float (between 0 and 1)
        Annealing factor for the global regularization (CPD). The factor is reduced every iteration by the annealing factor. 1 is no annealing.
    """

    def __init__(
        self,
        qInit,
        model,
        Y,
        q0=None,
        max_iterations=None,
        tolerance=None,
        lambdaFactor=None,
        damping=None,
        stiffness=None,
        gravity=None,
        sigma2=None,
        beta=None,
        mu=None,
        alpha=None,
        minDampingFactor=None,
        dampingAnnealing=None,
        stiffnessAnnealing=None,
        gravitationalAnnealing=None,
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

        if lambdaFactor is not None and (
            not isinstance(lambdaFactor, numbers.Number) or lambdaFactor < 0
        ):
            raise ValueError(
                "Expected a positive value for regularization parameter tau. Instead got: {}".format(
                    lambdaFactor
                )
            )

        if beta is not None and (not isinstance(beta, numbers.Number) or beta <= 0):
            raise ValueError(
                "Expected a positive value for the width of the coherent Gaussian kernel. Instead got: {}".format(
                    beta
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
        self.model = model

        self.Y = Y
        self.T = model.getPositions(self.q)
        self.Dof = self.q.size
        (self.N, _) = self.T.shape
        (self.M, self.D) = self.Y.shape
        self.tolerance = 10e-5 if tolerance is None else tolerance
        self.max_iterations = 100 if max_iterations is None else max_iterations
        self.iteration = 0
        self.damping = 1 if damping is None else damping
        self.stiffness = 1 if stiffness is None else stiffness
        self.gravity = np.array([0, 0, 0]) if gravity is None else gravity
        self.beta = 2 if beta is None else beta
        self.sigma2 = initialize_sigma2(self.T, self.Y) if sigma2 is None else sigma2
        self.mu = 0.0 if mu is None else mu
        self.alpha = 1 if alpha is None else alpha
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
        self.G = gaussian_kernel(self.T, self.beta)
        self.Gq = np.eye(self.Dof)
        # self.Gq = np.eye(self.Dof) - 0.3 * (
        #     np.ones((self.Dof, self.Dof)) - np.eye(self.Dof)
        # )

        # # decoupling the translational and rotational degrees of freedom
        # self.Gq[3:6, :] = np.zeros((3, self.Dof))
        # self.Gq[:, 3:6] = np.zeros((self.Dof, 3))
        # self.Gq[3:6, 3:6] = np.eye(3)
        # # decoupling the translational and rotational degrees of freedom

    def register(self, callback):
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

    def estimateCorrespondance(self):
        """
        E-step: Compute the expectation step  of the EM algorithm.
        """
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
            return

    def dampedPseudoInverse(self, J, dampingFactor):
        dim = J.shape[0]
        dampedPseudoInverse = J.T @ np.linalg.inv(
            J @ J.T + dampingFactor**2 * np.eye(dim)
        )
        return dampedPseudoInverse

    def updateParameters(self):
        """
        M-step: Calculate a new parameters of the registration.
        """

        jacobianDamping = (self.dampingAnnealing) ** (self.iteration) * self.damping
        if (
            jacobianDamping < self.minDampingFactor
        ):  # lower limit of regularization to ensure stability of matrix inversion
            jacobianDamping = self.minDampingFactor

        dP1 = np.diag(self.P1)

        # method 1: determine joint change from CPD velocities
        # A = np.dot(dP1, self.G) + lambdaFactor * self.sigma2 * np.eye(self.N)
        # B = self.PY - np.dot(dP1, self.computeTargets(self.q))
        # self.W = np.linalg.solve(A, B)
        # J = np.zeros((self.D * self.N, self.Dof))
        # WGT = np.zeros(self.D * self.N)
        # for n in range(0, self.N):
        #     tempJ = self.model.getJacobian(self.q, n)
        #     for d in range(0, self.D):
        #         J[n * self.D + d, :] = tempJ[d, :]
        #         WGT[n * self.D + d] = self.W.T[d, :] @ self.G[n, :].T
        # Jdamped = self.dampedPseudoInverse(J, jacobianDamping)
        # self.dq = Jdamped @ WGT
        # self.updateDegreesOfFreedom()

        # method 2: determine joint change from solving system of equations
        A = np.zeros((self.Dof, self.Dof))
        B = np.zeros(self.Dof)
        dEGrav_dq = np.zeros(self.Dof)

        for n in range(0, self.N):
            J = self.model.getJacobian(self.q, n)
            dEGrav_dq += self.gravity @ J
            for m in range(0, self.M):
                # A += self.P[n, m] * (self.Gq.T @ J.T @ J @ self.Gq)
                # B += self.P[n, m] * (self.Gq.T @ J.T @ (self.Y[m, :] - self.T[n, :]).T)
                A += self.alpha * self.P[n, m] * (J.T @ J)
                B += self.alpha * self.P[n, m] * (J.T @ (self.Y[m, :] - self.T[n, :]).T)
        A += (
            self.sigma2 * self.stiffness * np.eye(self.Dof)
        )  # add stiffness term for left side
        B += (
            self.sigma2 * self.stiffness * np.eye(self.Dof) @ (self.q0 - self.q)
        )  # add stiffness term for right side
        B += self.sigma2 * dEGrav_dq  # add gravitational term
        AInvDamped = self.dampedPseudoInverse(A, jacobianDamping)
        self.dq = AInvDamped @ B
        self.updateDegreesOfFreedom()

        # self.W = np.linalg.pinv(A) @ B
        # self.W = np.linalg.solve(A, B)
        # self.dq = np.linalg.lstsq(Jdamped, WGT)[0]

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
        self.G: numpy array
            Gaussian kernel matrix.
        self.W: numpy array
            weight matrix matrix.
        """
        return self.G, self.W
