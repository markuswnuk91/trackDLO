import numpy as np
import numbers
from warnings import warn


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
        self, X, Y, max_iterations=None, tolerance=None, normalize=0, *args, **kwargs
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

        self.X = X
        self.Y = Y
        self.T = X
        (self.N, self.D) = self.X.shape
        (self.M, _) = self.Y.shape
        self.tolerance = 10e-5 if tolerance is None else tolerance
        self.max_iterations = 100 if max_iterations is None else max_iterations
        self.iteration = 0
        self.normalize = bool(normalize) if normalize is None else bool(normalize)

    def register(self, callback=lambda **kwargs: None):
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
                # kwargs = {
                #     "iteration": self.iteration,
                #     "error": self.diff,
                #     "X": self.Y,
                #     "Y": self.T,
                # }
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
