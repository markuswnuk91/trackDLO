import os
import sys
import numpy as np
import time


# Custom Imports
try:

    sys.path.append(os.getcwd().replace("/src/robot/", ""))

    # from src.Robot.RobotInterface.build.python.panda_python_bindings import pyPanda
    from src.robot.build.pythonBindings.robotBindings import highLevelControl


except:
    print("Import for calss Franka failed")
    raise

class FrankaEmikaPanda(object):
        def __init__(self) -> None:
            self.robotInterface = highLevelControl()
        

        def getO_T_EE(self, verbose=False):
            state = self.robotInterface.getRobotState()
            if verbose:
                print("O_T_EE: ", repr(np.reshape(state.O_T_EE, (4, 4)).T))
                print("q: ", state.q)
                print("dq: ", state.dq)
            return np.reshape(state.O_T_EE, (4, 4)).T