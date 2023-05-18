import os, sys
import numpy as np

try:
    sys.path.append(os.getcwd().replace("/app", ""))
    from src.robot.franka import FrankaEmikaPanda
except:
    print("Imports for application printRobotPose failed.")
    raise

def printRobotPose():
    robot = FrankaEmikaPanda()
    T_EE = robot.getO_T_EE()
    np.set_printoptions(formatter={'float': '{:0.9f}'.format})
    print(repr(T_EE))

if __name__ == "__main__":
    printRobotPose()
