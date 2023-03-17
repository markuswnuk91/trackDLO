import os, sys
import numpy as np

try:
    sys.path.append(os.getcwd().replace("/src/simulation", ""))
    from src.modelling.topologyTemplates import topologyGraph_ArenaWireHarness
    from src.simulation.bdlo import BranchedDeformableLinearObject
except:
    print("Imports for BDLO Templates failed.")
    raise


def initArenaWireHarness():
    arenaWireHarnessModel = BranchedDeformableLinearObject(
        **{"adjacencyMatrix": topologyGraph_ArenaWireHarness}
    )
    arenaWireHarnessModel.setBranchRootDof(1, 0, np.pi * 3 / 4)
    arenaWireHarnessModel.setBranchRootDofs(2, np.array([0, 0, 0]))
    arenaWireHarnessModel.setBranchRootDofs(3, np.array([-np.pi * 3 / 4, 0, 0]))
    arenaWireHarnessModel.setBranchRootDofs(4, np.array([0, 0, 0]))
    arenaWireHarnessModel.setBranchRootDofs(5, np.array([np.pi / 4, 0, 0]))
    arenaWireHarnessModel.setBranchRootDofs(6, np.array([0, 0, 0]))

    return arenaWireHarnessModel
