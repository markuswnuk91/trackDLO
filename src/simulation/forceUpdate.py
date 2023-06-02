import numpy as np
import numbers
from warnings import warn
from src.utils.utils import initialize_sigma2


class ForceUpdate(object):
    """Class for caluclating the forces to update dart simulation

    Attributes:
    -------------
    """

    def __init__(self, dartSkel, Kp=None, Kd=None, *args, **kwargs):
        self.skel = dartSkel
        self.Kp = 1 if Kp is None else Kp
        self.Kd = 0.1 if Kd is None else Kd

    def updateSkeleton(self, q, q_dot, q_ddot):
        self.skel.setPositions(q)
        self.skel.setVelocities(q_dot)
        self.skel.setAccelerations(q_ddot)

    def computeExternalForceUpdateInGeneralizedCoordinates(
        self,
        q,
        q_dot,
        q_ddot,
        qd,
        qd_dot,
        qd_ddot,
        method="PD",
    ):
        # update skeleton
        self.updateSkeleton(q, q_dot, q_ddot)
        # compute forces
        if method == "FeedbackLinearization":
            # reference : Shiyu Jin et al., Real-time State Estimation of Deformable Objects with Dynamical Simulation, IROS, 2020
            tau = (
                self.skel.getMassMatrix() @ qd_ddot
                + self.skel.getCoriolisAndGravityForces()
                + self.Kp * (qd - q)
                + self.Kd * (qd_dot - q_dot)
            )
        if method == "PD":
            # reference : Shiyu Jin et al., Real-time State Estimation of Deformable Objects with Dynamical Simulation, IROS, 2020
            tau = self.Kp * (qd - q) - self.Kd * (q_dot)

        return tau
