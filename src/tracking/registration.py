import numpy as np
import numbers
from warnings import warn
from src.utils.utils import initialize_sigma2
import time


class NonRigidRegistration(object):
    """Base class for non-rigid registration for DLO

    Attributes:
    -------------
    X: numpy array
        NxD array of source points

    Y: numpy array
        MxD array of data points (e.g. 3D point cloud)

    T: numpy array
        NxD array of target points (transformed source points)

    N: int
        Number of source points

    M: int
        Number of data points

    D: int
        Dimensionality of source and target points

    iterations: int
        The current iteration throughout the registration

    max_iterations: int
        Maximum number of iterations the registration performs before terminating

    tolerance: float (positive)
        tolerance for checking convergence.
        Registration will terminate once the difference between
        consecutive objective function values falls within this tolerance.
    """

    def __init__(
        self,
        X,
        Y,
        max_iterations=None,
        tolerance=None,
        sigma2=None,
        mu=None,
        normalize=0,
        logging=False,
        *args,
        **kwargs
    ):
        if type(X) is not np.ndarray or X.ndim != 2:
            raise ValueError("The source point cloud (X) must be at a 2D numpy array.")

        if type(Y) is not np.ndarray or Y.ndim != 2:
            raise ValueError("The target point cloud (Y) must be a 2D numpy array.")

        if X.shape[1] != Y.shape[1]:
            raise ValueError(
                "Both point clouds need to have the same number of dimensions."
            )
        if X.shape[0] < X.shape[1] or Y.shape[0] < Y.shape[1]:
            raise ValueError(
                "The dimensionality is larger than the number of points. Possibly the wrong orientation of X and Y."
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
        self.X = X
        self.Y = Y
        self.T = X
        (self.N, self.D) = self.X.shape
        (self.M, _) = self.Y.shape
        self.tolerance = 10e-5 if tolerance is None else tolerance
        self.max_iterations = 100 if max_iterations is None else max_iterations
        self.iteration = 0
        self.sigma2 = initialize_sigma2(self.X, self.Y) if sigma2 is None else sigma2
        self.mu = 0.0 if mu is None else mu
        self.normalize = bool(normalize) if normalize is None else bool(normalize)
        self.callback = None
        self.totalIterations = 0
        # runtime counter initializaton
        self.runTimes = {}
        self.runTimes["runtimesPerIteration"] = []
        self.runTimes["correspondanceEstimation"] = []
        self.runTimes["parameterUpdate"] = []
        self.runTimes["targetComputation"] = []
        self.logging = logging
        if self.logging:
            self.log = {
                "X": [self.X],
                "Y": [self.Y],
                "T": [self.T],
                "iteration": [self.iteration],
            }

    def register(self, checkConvergence=True, customCallback=None, *args, **kwargs):
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
        self.iteration = 0
        self.runTimes["runtimesPerRegistration"] = []
        # self.computeTargets()
        run = True
        while run:
            if self.iteration >= self.max_iterations:
                run = False
            if checkConvergence and self.isConverged():
                run = False
            if run:
                runtimePerIteration_start = time.time()
                self.iterate()
                runtimePerIteration_end = time.time()
                runtimePerIteration = (
                    runtimePerIteration_end - runtimePerIteration_start
                )
                self.runTimes["runtimesPerIteration"].append(runtimePerIteration)
                self.runTimes["runtimesPerRegistration"].append(runtimePerIteration)
                if callable(self.callback):
                    self.callback()
                if callable(customCallback):
                    customCallback(*args, **kwargs)
        return self.T, self.getParameters()

    def iterate(self):
        """
        Perform one iteration of the registration.
        """
        correspondanceEstimationRuntime_start = time.time()
        self.estimateCorrespondance()
        correspondanceEstimationRuntime_end = time.time()
        self.runTimes["correspondanceEstimation"].append(
            correspondanceEstimationRuntime_end - correspondanceEstimationRuntime_start
        )

        parameterUpdateRuntime_start = time.time()
        self.updateParameters()
        parameterUpdateRuntime_end = time.time()
        self.runTimes["parameterUpdate"].append(
            parameterUpdateRuntime_end - parameterUpdateRuntime_start
        )

        targetComputationRuntime_start = time.time()
        self.computeTargets()
        targetComputationRuntime_end = time.time()
        self.runTimes["targetComputation"].append(
            targetComputationRuntime_end - targetComputationRuntime_start
        )

        self.iteration += 1
        self.totalIterations += 1

    def isConverged(self):
        """
        Placeholder for child classes.
        """
        raise NotImplementedError(
            "Checking for convergence should be defined in child classes."
        )

    def estimateCorrespondance(self):
        """
        E-step: Compute the expectation step  of the EM algorithm.
        """
        if self.normalize:
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

    def computeTargets(self):
        """
        Placeholder for child classes.
        """
        raise NotImplementedError(
            "Updating the source points should be defined in child classes."
        )

    def updateParameters(self):
        """
        Placeholder for child classes.
        """
        raise NotImplementedError(
            "Updates of parameters should be defined in child classes."
        )

    def getParameters(self):
        """
        Placeholder for child classes.
        """
        raise NotImplementedError(
            "Registration parameters should be defined in child classes."
        )

    def registerCallback(self, callback):
        self.callback = callback

    def setTargetPointCloud(self, Y):
        (self.M, _) = Y.shape
        self.Y = Y
        self.X = self.T
        self.reinitializeParameters()

    def initializeParameters(self):
        """
        Placeholder for child classes.
        """
        raise NotImplementedError(
            "Updating the source points should be defined in child classes."
        )

    def reinitializeParameters(self):
        """
        Placeholder for child classes.
        """
        raise NotImplementedError(
            "Updating the source points should be defined in child classes."
        )
