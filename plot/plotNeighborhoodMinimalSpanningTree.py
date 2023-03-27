import os
import sys
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np


try:
    sys.path.append(os.getcwd().replace("/tests", ""))

    from src.visualization.curveShapes3D import helixShape
    from src.visualization.plot3D import (
        plotPointSets,
        plotPointSetAsLine,
        plotPointSetsAsLine,
    )
    from src.sensing.loadPointCloud import readPointCloudFromPLY
except:
    print("Imports for Neighborhood MST failed.")
    raise
