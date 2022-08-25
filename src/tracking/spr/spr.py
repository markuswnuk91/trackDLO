from builtins import super
import os
import sys
import numpy as np

try:
    sys.path.append(os.getcwd().replace("/src/tracking/spr", ""))
    from src.tracking.registration import NonRigidRegistration
    from src.tracking.utils.utils import gaussian_kernel, initialize_sigma2
except:
    print("Imports for CPD failed.")
    raise

class StructurePreservedRegistration(NonRigidRegistration):
    """
    Implementation of the Structure Preserved Registration (SPR) according to
    the paper: 
    Tang, T. and Tomizuka, M. (2022); "Track deformable objects from point clouds with structure preserved registration", The International Journal of Robotics Research, 41(6), pp. 599–614. doi: 10.1177/0278364919841431.
    Based on their provided Matlab implementation:
    https://github.com/thomastangucb/SPR

    Attributes
    ----------
    tau: float (positive)
        Regularization factor for the Local Regularization (MLLE).
        A higher tau forces the enforces stronger local structure perservation between points.

    beta: float(positive)
        Width of the Gaussian kernel for global regularization.
        A higher beta enforces a stronger coherence in the movement of the points, the points behave "more stiff".

    sigma2: float (positive)
        Variance of the Gaussian mixture model

    diff: float (positive)
        The absolute difference between the current and previous objective function values.

    P: numpy array
        MxN array of probabilities.
        P[m, n] represents the probability that the m-th source point
        corresponds to the n-th target point.

    mu: float (between 0 and 1)
        Contribution of the uniform distribution to account for outliers.
        Valid values span 0 (inclusive) and 1 (exclusive).
    """
    def __init__(self, tau = None, beta=None, sigma2=None, mu=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if tau is not None and (not isinstance(tau, numbers.Number) or tau <= 0):
                    raise ValueError(
                        "Expected a positive value for regularization parameter tau. Instead got: {}".format(tau))

        self.tau = 3 if tau is None else tau
        self.beta = 2 if beta is None else beta
        self.sigma2 = initialize_sigma2(self.X,self.Y) if sigma2 is None else sigma2