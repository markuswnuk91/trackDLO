import os, sys
import dartpy as dart
import numpy as np

try:
    sys.path.append(os.getcwd().replace("/tests", ""))
    from src.visualization.dartVisualizer import DartVisualizer
    from src.simulation.dlo import DeformableLinearObject
except:
    print("Imports for testing DartVisualizer Loader failed.")
    raise


def testDartVis():
    dartVis = DartVisualizer()
    dlo = DeformableLinearObject(30)
    dartVis.addSkeleton(dlo.skel)
    dartVis.run()


def testForceIK():
    dartVis = DartVisualizer()
    numBodies = 30
    dlo = DeformableLinearObject(numBodies)
    currentPositions = dlo.computeForwardKinematics(dlo.getPositions())
    targetPositions = currentPositions + np.array(([1, 0, 0]))
    dartVis.addSkeleton(dlo.skel)
    q = dartVis.computeInverseKinematicsWithCartesianForces(
        dlo.skel, targetPositions, vis=False, verbose=False
    )
    print(q)


if __name__ == "__main__":
    # testDartVis()
    testForceIK()
