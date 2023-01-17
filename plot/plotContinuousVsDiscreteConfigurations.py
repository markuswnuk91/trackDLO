# plots different configurations of a DLO
# figure in chapter 4, Discretization

import os
import sys
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import numpy as np

try:
    sys.path.append(os.getcwd().replace("/tests", ""))
    from src.modelling.wakamatsuModel import (
        WakamatsuModel,
    )
    from src.modelling.utils.utils import (
        loadWakamatsuModelFromJson,
    )
    from src.visualization.plot3D import (
        setupLatexPlot3D,
        plotPointSet,
        plotPointSetAsLine,
        plotPointSetAsColorGradedLine,
    )
except:
    print("Imports for plotting continuous vs discrete approximations failed.")
    raise

# plot control
saveFigs = True
savePath = "/mnt/c/Users/ac129490/Documents/Dissertation/Thesis/62bebc3388a16f7dcc7f9153/figures/"

loadPathContinuousConfig = "/mnt/c/Users/ac129490/Documents/Dissertation/Software/trackdlo/plot/plotdata/helixReconstruction/helix_continuousModel.json"


if __name__ == "__main__":

    continuousModel = loadWakamatsuModelFromJson(loadPathContinuousConfig)

    # plotting
    fig, ax = setupLatexPlot3D()
    # plot continuous config
    plotPointSetAsLine(
        ax=ax, X=continuousModel.evalPositions(np.linspace(0, continuousModel.L, 100))
    )
