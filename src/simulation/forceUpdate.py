import numpy as np
import numbers
from warnings import warn
from src.utils.utils import initialize_sigma2


class ForceUpdate(object):
    """Class for caluclating the forces to update dart simulation

    Attributes:
    -------------
    """

    def __init__(self, dartSkel, Kp=None, Kd=None, forceLimit=None, *args, **kwargs):
        self.skel = dartSkel
        self.Kp = 1 if Kp is None else Kp
        self.Kd = 0.1 if Kd is None else Kd
        self.forceLimit = 0.1 if forceLimit is None else forceLimit
        self.velocityLimit = np.pi / 10

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
        q_old,
        skel,
        method="StablePD",  #  PD, FeedbackLinearization, StablePD,
    ):
        # update skeleton
        # self.updateSkeleton(q, q_dot, q_ddot)
        # compute forces
        # if np.any(np.abs(q_dot) > self.velocityLimit):
        #     for i, vel in enumerate(q_dot):
        #         q_dot[i] = np.sign(vel) * self.velocityLimit

        if method == "FeedbackLinearization":
            # reference : Shiyu Jin et al., Real-time State Estimation of Deformable Objects with Dynamical Simulation, IROS, 2020
            # q += q_dot * self.skel.getTimeStep()
            qd_ddot += (
                skel.getInvMassMatrix()
                @ (-skel.getCoriolisAndGravityForces() + skel.getExternalForces())
                * skel.getTimeStep()
            )
            tau = (
                +skel.getMassMatrix() @ qd_ddot
                + skel.getCoriolisAndGravityForces()
                + self.Kp * (qd - skel.getPositions())
                + self.Kd * (-skel.getVelocities())
            )
        if method == "PD":
            tau = self.Kp * (qd - q) - self.Kd * q_dot

        if method == "StablePD":
            q = skel.getPositions()
            q_dot = skel.getVelocities()
            delta_t = skel.getTimeStep()
            M = skel.getMassMatrix()
            C = skel.getCoriolisForces()
            g = skel.getGravityForces()
            tau_ext = skel.getExternalForces()
            q_ddot = np.linalg.inv(M + self.Kd * np.eye(len(q)) * delta_t) @ (
                -C - g + self.Kp * (qd - q - q_dot * delta_t) - self.Kd * q_dot
            )
            tau = self.Kp * (qd - q - q_dot * delta_t) - self.Kd * (
                q_dot + q_ddot * delta_t
            )
            # q = skel.getPositions() + skel.getVelocities() * self.skel.getTimeStep()
            # qError = qd - q
            # dqError = -(
            #     skel.getVelocities() + skel.getTimeStep() * skel.getAccelerations()
            # )
            # # dqError = -skel.getVelocities()
            # M = skel.getMassMatrix()
            # Cg = skel.getCoriolisAndGravityForces()
            # # Cg = skel.getCoriolisForces()
            # tau = np.zeros((skel.getNumDofs()))
            # tau = (
            #     M @ (self.Kp * qError + self.Kd * dqError)
            #     + Cg
            #     # + skel.getMassMatrix() @ skel.getAccelerations()
            # )

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
        # if np.any(np.abs(tau) > self.forceLimit):
        #     indices = np.where(np.abs(tau) > self.forceLimit)[0]
        #     for index in indices:
        #         tau[index] = np.sign(tau[index]) * self.forceLimit
        return tau
