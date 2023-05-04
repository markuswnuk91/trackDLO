import os
import sys
import numpy as np
import time


# Custom Imports
try:

    sys.path.append(os.getcwd().replace("/src/robot/", ""))

    # from src.Robot.RobotInterface.build.python.panda_python_bindings import pyPanda
    from src.robot.build.pythonBindings import libfrankaInterface


except:
    print("Import for calss Franka failed")
    raise

class FrankaEmikaPanda(object):
        def __init__(self) -> None:
            self.robot = libfrankaInterface.Robot("172.16.0.2")
        

        def getO_T_EE(self, verbose=False):
            state = self.robot.readOnce()
            if verbose:
                print("O_T_EE: ", repr(np.reshape(state.O_T_EE, (4, 4)).T))
                print("q: ", state.q)
                print("dq: ", state.dq)
            return np.reshape(state.O_T_EE, (4, 4)).T
        
        def getRobotState(self):
            roboState = self.robot.readOnce()
            roboDict = {}
            roboDict["O_T_EE"] = roboState.O_T_EE
            roboDict["O_T_EE_d"] = roboState.O_T_EE_d
            roboDict["F_T_EE"] = roboState.F_T_EE
            roboDict["F_T_NE"] = roboState.F_T_NE
            roboDict["NE_T_EE"] = roboState.NE_T_EE
            roboDict["EE_T_K"] = roboState.EE_T_K
            roboDict["m_ee"] = roboState.m_ee
            roboDict["I_ee"] = roboState.I_ee
            roboDict["F_x_Cee"] = roboState.F_x_Cee
            roboDict["m_load"] = roboState.m_load
            roboDict["I_load"] = roboState.I_load
            roboDict["F_x_Cload"] = roboState.F_x_Cload
            roboDict["m_total"] = roboState.m_total
            roboDict["I_total"] = roboState.I_total
            roboDict["F_x_Ctotal"] = roboState.F_x_Ctotal
            roboDict["elbow"] = roboState.elbow
            roboDict["elbow_d"] = roboState.elbow_d
            roboDict["elbow_c"] = roboState.elbow_c
            roboDict["delbow_c"] = roboState.delbow_c
            roboDict["ddelbow_c"] = roboState.ddelbow_c
            roboDict["tau_J"] = roboState.tau_J
            roboDict["tau_J_d"] = roboState.tau_J_d
            roboDict["dtau_J"] = roboState.dtau_J
            roboDict["q"] = roboState.q
            roboDict["q_d"] = roboState.q_d
            roboDict["dq"] = roboState.dq
            roboDict["dq_d"] = roboState.dq_d
            roboDict["ddq_d"] = roboState.ddq_d
            roboDict["joint_contact"] = roboState.joint_contact
            roboDict["cartesian_contact"] = roboState.cartesian_contact
            roboDict["joint_collision"] = roboState.joint_collision
            roboDict["cartesian_collision"] = roboState.cartesian_collision
            roboDict["tau_ext_hat_filtered"] = roboState.tau_ext_hat_filtered
            roboDict["O_F_ext_hat_K"] = roboState.O_F_ext_hat_K
            roboDict["K_F_ext_hat_K"] = roboState.K_F_ext_hat_K
            roboDict["O_dP_EE_d"] = roboState.O_dP_EE_d
            roboDict["O_T_EE_c"] = roboState.O_T_EE_c
            roboDict["O_dP_EE_c"] = roboState.O_dP_EE_c
            roboDict["O_ddP_EE_c"] = roboState.O_ddP_EE_c
            roboDict["theta"] = roboState.theta
            roboDict["dtheta"] = roboState.dtheta
            roboDict["time"] = roboState.time.toSec()
            return roboDict
