import os, sys

try:
    sys.path.append(os.getcwd().replace("/tests", ""))
    from src.robot.franka import FrankaEmikaPanda
except:
    print("Imports for BDLO testing failed.")
    raise

def test_printEEPose():
    robot = FrankaEmikaPanda()
    while True:
        print(robot.getO_T_EE())

def test_printRobotState():
    robot = FrankaEmikaPanda()
    while True:
        print(robot.getRobotState())

if __name__ == "__main__":
    # test_printEEPose()
    test_printRobotState()
