import os, sys
import numpy as np

try:
    sys.path.append(os.getcwd().replace("/app", ""))
    from src.robot.franka import FrankaEmikaPanda
except:
    print("Imports for application closeRobotGripper failed.")
    raise

width = 0.02

def closeRobotGripper(width):
    robot = FrankaEmikaPanda()
    robot.moveGripper(width)

if __name__ == "__main__":
    closeRobotGripper(width)