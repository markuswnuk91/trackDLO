import sys, os
import matplotlib.pyplot as plt
import numpy as np

try:
    sys.path.append(os.getcwd().replace("/src/visualization", ""))
    from src.visualization.plotUtils import set_size, set_axes_equal
except:
    print("Imports for Plot2D File failed.")
    raise


def setupLatexPlot2D(
    figureWidth=483.6969,
    figureHeight=None,
    axisLimX=[0, 1],
    axisLimY=[0, 1],
    xlabel="$x$",
    ylabel="$y$",
    xTickStep=None,
    yTickStep=None,
):
    if figureHeight is not None:
        fig = plt.figure(figsize=set_size(width=figureWidth, height=figureHeight))
    else:
        fig = plt.figure(figsize=set_size(width=figureWidth))
    ax = fig.add_subplot()

    # set axis limits
    ax.set_xlim(axisLimX[0], axisLimX[1])
    ax.set_ylim(axisLimY[0], axisLimY[1])

    # set axis lables
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # set x ticks
    if xTickStep is not None:
        ax.set_xticks(np.arange(axisLimX[0], axisLimX[1] + xTickStep, step=xTickStep))
    if yTickStep is not None:
        ax.set_yticks(np.arange(axisLimY[0], axisLimY[1] + yTickStep, step=yTickStep))
    return fig, ax


def plotImage(rgbImage):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(w, h)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(rgbImage, aspect="auto")
