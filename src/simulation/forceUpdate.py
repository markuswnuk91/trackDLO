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
        skel,
        method="StablePD",  #  PD, FeedbackLinearization, StablePD,
    ):
        # update skeleton
        # self.updateSkeleton(q, q_dot, q_ddot)
        # compute forces
        if method == "FeedbackLinearization":
            # reference : Shiyu Jin et al., Real-time State Estimation of Deformable Objects with Dynamical Simulation, IROS, 2020
            # q += q_dot * self.skel.getTimeStep()
            tau = (
                skel.getMassMatrix() @ skel.getAccelerations()
                + skel.getCoriolisAndGravityForces()
                + self.Kp * (qd - skel.getPositions())
                + self.Kd * (qd_dot - skel.getVelocities())
            )
        if method == "PD":
            tau = self.Kp * (qd - q) - self.Kd * (q_dot)

        if method == "StablePD":
            q = skel.getPositions()
            q += skel.getVelocities() * self.skel.getTimeStep()
            qError = qd - q
            dqError = -skel.getVelocities()
            M = skel.getMassMatrix()
            Cg = skel.getCoriolisAndGravityForces()
            tau = np.zeros((skel.getNumDofs()))
            tau = M @ (self.Kp * qError + self.Kd * dqError) + Cg
            # # cartesian controller
            # cartesianError = qd[3:6] - q[3:6]
            # cartesianVelocityError = qd_dot[3:6] - q_dot[3:6]
            # tau[3:6] = self.skel.getMass() * (
            #     self.Kp * qError[3:6] + self.Kd * dqError[3:6]
            # )

            # # joint space controller
            # tau[6:] = (
            #     M[6:, 6:] @ (self.Kp * qError[6:] + self.Kd * dqError[6:]) + Cg[6:]
            # )
        return tau
