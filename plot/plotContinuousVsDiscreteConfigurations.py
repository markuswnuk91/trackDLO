# plots different configurations of a DLO
# figure in chapter 4, Discretization

import os
import sys
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
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
    from src.modelling.utils.calculateArcLength import calcArcLengthFromCurveFun
    from src.visualization.curveShapes3D import helixShape
    from src.visualization.plot3D import (
        plotPointSets,
        plotPointSetAsLine,
        plotPointSetsAsLine,
    )
    from src.reconstruction.continuousReconstruction import ContinuousReconstruction
    from src.reconstruction.discreteReconstruction import DiscreteReconstruction
    from src.modelling.utils.utils import (
        loadWakamatsuModelParametersFromJson,
    )
except:
    print("Imports for plotting continuous vs discrete approximations failed.")
    raise

# plot control
saveFigs = False
savePath = "/mnt/c/Users/ac129490/Documents/Dissertation/Thesis/62bebc3388a16f7dcc7f9153/figures/"
saveNames = [
    "DiscreteModelConfiguration_Coarse.pgf",
    "DiscreteModelConfiguration_Fine.pgf",
]  # requrires as many entries as discretizations

loadPathContinuousConfig = "/mnt/c/Users/ac129490/Documents/Dissertation/Software/trackdlo/plot/plotdata/helixReconstruction/helix_continuousModel.json"

colorMap = matplotlib.colormaps["viridis"]
textwidth_in_pt = 483.6969
figureScaling = 0.4
latexFontSize_in_pt = 14
latexFootNoteFontSize_in_pt = 10
desiredFigureWidth = figureScaling * textwidth_in_pt
desiredFigureHeight = figureScaling * textwidth_in_pt
tex_fonts = {
    #    "pgf.texsystem": "pdflatex",
    # Use LaTeX to write all text
    "text.usetex": True,
    "font.family": "serif",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": latexFontSize_in_pt,
    "font.size": latexFontSize_in_pt,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": latexFootNoteFontSize_in_pt,
    "xtick.labelsize": latexFootNoteFontSize_in_pt,
    "ytick.labelsize": latexFootNoteFontSize_in_pt,
}
if saveFigs:
    matplotlib.use("pgf")
    matplotlib.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "font.family": "serif",
            "text.usetex": True,
            "pgf.rcfonts": False,
        }
    )
    matplotlib.rcParams.update(tex_fonts)

# discrete reconsruction
reconstructDiscrete = True
numIterDiscrete = 70
discretizations = [5, 10]

if __name__ == "__main__":
    if len(discretizations) != len(saveNames):
        raise ValueError(
            "Expeted a individual save path for every discretization. Check if enough savePaths have been specified."
        )

    for i, numSegments in enumerate(discretizations):
        # helix definition & reconstuction
        helixCurve = lambda s: helixShape(s, heightScaling=2.0, frequency=2)
        arcLength = calcArcLengthFromCurveFun(helixCurve, 0, 1)

        s = np.linspace(0, 1, 1000)
        Y = helixCurve(s)

        sDiscrete = np.linspace(0, 1, numSegments + 1)
        discreteY = helixCurve(sDiscrete)

        # # plotting
        fig, ax = setupLatexPlot3D(
            figureWidth=desiredFigureWidth,
            figureHeight=desiredFigureHeight,
            axisLimX=[-1, 1],
            axisLimY=[-1, 1],
            axisLimZ=[0, 2],
            viewAngle=(10, 30),
            xTickStep=0.5,
            yTickStep=0.5,
            zTickStep=0.5,
        )
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        plotPointSetsAsLine(
            ax=ax,
            X=discreteY,
            # Y=continuousModel.evalPositions(np.linspace(0, continuousModel.L, 100)),
            Y=Y,
            waitTime=-1,
            linewidthX=3.0,
            linewidthY=2.0,
            alphaX=0.8,
            alphaY=0.8,
        )
        # create legend
        redLine = mlines.Line2D([], [], color=[1, 0, 0], label="continuous")
        blueDottedLine = mlines.Line2D(
            [], [], color=[0, 0, 1], marker=".", markersize=10, label="discrete"
        )
        plt.legend(
            handles=[redLine, blueDottedLine],
            loc="upper right",
            bbox_to_anchor=(1, 0.8),
        )
        plotPointSet(ax=ax, X=discreteY, waitTime=None, size=80, alpha=0.5)

        # save figures
        if saveFigs:
            plt.savefig(
                savePath + saveNames[i],
                bbox_inches="tight",
            )
