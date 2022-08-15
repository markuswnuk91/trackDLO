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
        Dimensionality of source and target poitns

    C: numpy array
        Correspondance vector such that X corresponds to Y(C,:)

    iterations: int
        The current iteration throughout the registration

    max_iterations: int
        Maximum number of iterations the registration performs before terminating

    tolerance: float (positive)
        tolerance for 

    vis: bool
        visualization for every iteration
    """

    def __init__(self, X, Y, max_iterations=None, tolerance=None, *args, **kwargs):
            if type(X) is not np.ndarray or X.ndim != 2:    
                raise ValueError(
                    "The target point cloud (X) must be at a 2D numpy array.")
                    
            if type(Y) is not np.ndarray or Y.ndim != 2:
                raise ValueError(
                    "The source point cloud (Y) must be a 2D numpy array.")

            if X.shape[1] != Y.shape[1]:
                raise ValueError(
                    "Both point clouds need to have the same number of dimensions.")

            if max_iterations is not None and (not isinstance(max_iterations, numbers.Number) or max_iterations < 0):
                raise ValueError(
                    "Expected a positive integer for max_iterations instead got: {}".format(max_iterations))
            elif isinstance(max_iterations, numbers.Number) and not isinstance(max_iterations, int):
                warn("Received a non-integer value for max_iterations: {}. Casting to integer.".format(max_iterations))
                max_iterations = int(max_iterations)

            if tolerance is not None and (not isinstance(tolerance, numbers.Number) or tolerance < 0):
                raise ValueError(
                    "Expected a positive float for tolerance instead got: {}".format(tolerance))

            self.X = X
            self.Y = Y
            self.T = X
            (self.N, self.D) = self.X.shape
            (self.M, _) = self.Y.shape
            self.tolerance = 10e-5 if tolerance is None else tolerance
            self.max_iterations = 100 if max_iterations is None else max_iterations
            self.iteration = 0

    def register(self, callback=lambda **kwargs:None):
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
                kwargs = {'iteration': self.iteration,
                           'error':self.diff, 'X': self.X, 'Y': self.T}
                callback(**kwargs)

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
            "Checking for convergence should be defined in child classes.")

    def estimateCorrespondance(self):
        """Placeholder for child class.
        """
        raise NotImplementedError(
            "Estimating the correspondance should be defined in child classes.")

    def computeTargets(self):
        """
        Placeholder for child classes.
        """
        raise NotImplementedError(
            "Updating the source points should be defined in child classes.")

    def updateParameters(self):
        """
        Placeholder for child classes.
        """
        raise NotImplementedError(
            "Updates of parameters should be defined in child classes.")

    def getParameters(self):
        """
        Placeholder for child classes.
        """
        raise NotImplementedError(
            "Registration parameters should be defined in child classes.")