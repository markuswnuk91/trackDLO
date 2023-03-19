import os
import sys
import numpy as np

try:
    sys.path.append(os.getcwd().replace("/tests", ""))
    from src.reconstruction.bdloReconstruction import (
        BDLOReconstruction,
    )
    from src.localization.correspondanceEstimation.topologyBasedCorrespondanceEstimation import (
        TopologyBasedCorrespondanceEstimation,
    )
    from src.localization.topologyExtraction.minimalSpanningTreeTopology import (
        MinimalSpanningTreeTopology,
    )
    from src.simulation.bdloTemplates import initArenaWireHarness
    from src.visualization.plot3D import (
        plotPointSets,
    )
except:
    print("Imports for BDLOReconstruction Test failed.")
    raise


vis = True  # enable for visualization
dataPath = "tests/testdata/topologyExtraction/wireHarnessReduced.txt"


def runReconstruction():
    testPointSet = np.loadtxt(dataPath)
    testBDLO = initArenaWireHarness()
    testCorrespondanceEstimator = TopologyBasedCorrespondanceEstimation(
        **{
            "Y": testPointSet,
            "extractedTopology": MinimalSpanningTreeTopology(testPointSet),
            "numSeedPoints": 70,
            "templateTopology": testBDLO,
        }
    )
    Y = testPointSet
    (
        CBY,
        SY,
    ) = testCorrespondanceEstimator.calculateBranchCorresponanceAndLocalCoordinatsForPointSet(
        Y
    )
    testReconstruction = BDLOReconstruction(
        **{"bdlo": testBDLO, "Y": Y, "SY": SY, "CBY": CBY}
    )
    testReconstruction.reconstructShape(numIter=100)


if __name__ == "__main__":
    runReconstruction()
