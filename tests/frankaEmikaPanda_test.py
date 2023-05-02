import os, sys

try:
    sys.path.append(os.getcwd().replace("/tests", ""))
    from src.robot.franka import FrankaEmikaPanda
except:
    print("Imports for BDLO testing failed.")
    raise



if __name__ == "__main__":
    robot = FrankaEmikaPanda()
    
    while True:
        print(robot.getO_T_EE())
