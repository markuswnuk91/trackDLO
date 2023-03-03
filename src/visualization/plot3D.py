import matplotlib.pyplot as plt
import numpy as np


def set_size(width, height=None, fraction=1, subplots=(1, 1)):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == "thesis":
        width_pt = 426.79135
    elif width == "beamer":
        width_pt = 307.28987
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    if height is None:
        fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])
    else:
        fig_height_in = height * inches_per_pt
    return (fig_width_in, fig_height_in)


def setupLatexPlot3D(
    figureWidth=483.6969,
    figureHeight=None,
    axisLimX=[0, 1],
    axisLimY=[0, 1],
    axisLimZ=[0, 1],
    xlabel="$x$",
    ylabel="$y$",
    zlabel="$z$",
    viewAngle=(0, 0),
    xTickStep=None,
    yTickStep=None,
    zTickStep=None,
):
    if figureHeight is not None:
        fig = plt.figure(figsize=set_size(width=figureWidth, height=figureHeight))
    else:
        fig = plt.figure(figsize=set_size(width=figureWidth))
    ax = fig.add_subplot(projection="3d")

    # set view angle
    ax.view_init(viewAngle[0], viewAngle[1])

    # set axis limits
    ax.set_xlim(axisLimX[0], axisLimX[1])
    ax.set_ylim(axisLimY[0], axisLimY[1])
    ax.set_zlim(axisLimZ[0], axisLimZ[1])

    # set axis lables
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)

    # set background color as white
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # set x ticks
    if xTickStep is not None:
        ax.set_xticks(np.arange(axisLimX[0], axisLimX[1] + xTickStep, step=xTickStep))
    if yTickStep is not None:
        ax.set_yticks(np.arange(axisLimY[0], axisLimY[1] + yTickStep, step=yTickStep))
    if zTickStep is not None:
        ax.set_zticks(np.arange(axisLimZ[0], axisLimZ[1] + zTickStep, step=zTickStep))
    return fig, ax


def plotPoint(
    x,
    ax,
    color=[0, 0, 1],
    alpha=1,
    label: str = None,
    size=20,
    waitTime=0.001,
):
    if label is None:
        ax.scatter(x[0], x[1], x[2], color=color, alpha=alpha, s=size)
    else:
        ax.scatter(
            x[0],
            x[1],
            x[2],
            color=color,
            label=label,
            alpha=alpha,
            s=size,
            edgecolor=None,
        )
    if waitTime is not None and waitTime != -1:
        plt.draw()
        plt.pause(waitTime)
    elif waitTime is None:
        plt.show(block=True)
    elif waitTime == -1:
        plt.show(block=False)


def plotPointSet(
    X,
    ax,
    color=[0, 0, 1],
    edgeColor=None,
    size=None,
    markerStyle=None,
    alpha=None,
    label: str = None,
    waitTime=0.001,
):
    size = 20 if size is None else size
    markerStyle = "o" if markerStyle is None else markerStyle
    alpha = 1 if alpha is None else alpha

    if label is None:
        ax.scatter(
            X[:, 0],
            X[:, 1],
            X[:, 2],
            color=color,
            alpha=alpha,
            s=size,
            marker=markerStyle,
            edgeColor=edgeColor,
        )
    else:
        ax.scatter(
            X[:, 0],
            X[:, 1],
            X[:, 2],
            color=color,
            label=label,
            alpha=alpha,
            s=size,
            marker=markerStyle,
            edgecolor=None,
        )
    if waitTime is not None and waitTime != -1:
        plt.draw()
        plt.pause(waitTime)
    elif waitTime is None:
        plt.show(block=True)
    elif waitTime == -1:
        plt.show(block=False)


def plotPointSets(
    ax,
    X,
    Y,
    xColor=[0, 0, 1],
    yColor=[1, 0, 0],
    xEdgeColor=None,
    yEdgeColor=None,
    xSize=None,
    ySize=None,
    xMarkerStyle=None,
    yMarkerStyle=None,
    yAlpha=None,
    xAlpha=None,
    xLabel=None,
    yLabel=None,
    waitTime=0.001,
):

    xSize = 20 if xSize is None else xSize
    ySize = 20 if xSize is None else xSize
    xMarkerStyle = "o" if xMarkerStyle is None else xMarkerStyle
    yMarkerStyle = "o" if yMarkerStyle is None else yMarkerStyle
    xAlpha = 1 if xAlpha is None else xAlpha
    yAlpha = 1 if yAlpha is None else yAlpha

    if xLabel is not None:
        ax.scatter(
            X[:, 0],
            X[:, 1],
            X[:, 2],
            s=xSize,
            color=xColor,
            marker=xMarkerStyle,
            alpha=xAlpha,
            label=xLabel,
        )
    else:
        ax.scatter(
            X[:, 0],
            X[:, 1],
            X[:, 2],
            s=xSize,
            color=xColor,
            marker=xMarkerStyle,
            alpha=xAlpha,
        )

    if yLabel is not None:
        ax.scatter(
            Y[:, 0],
            Y[:, 1],
            Y[:, 2],
            s=ySize,
            color=yColor,
            marker=yMarkerStyle,
            alpha=yAlpha,
            label=yLabel,
        )
    else:
        ax.scatter(
            Y[:, 0],
            Y[:, 1],
            Y[:, 2],
            s=ySize,
            color=yColor,
            marker=yMarkerStyle,
            alpha=yAlpha,
        )

    if waitTime is not None and waitTime != -1:
        plt.draw()
        plt.pause(waitTime)
    elif waitTime is None:
        plt.show(block=True)
    elif waitTime == -1:
        plt.show(block=False)


def plotPointSetAsLine(
    ax: plt.axis,
    X: np.array,
    label: str = None,
    color=[0, 0, 1],
    alpha=1,
    linewidth=1.5,
    waitTime=None,
):
    if label is None:
        ax.plot3D(
            X[:, 0], X[:, 1], X[:, 2], color=color, alpha=alpha, linewidth=linewidth
        )
    else:
        ax.plot3D(
            X[:, 0],
            X[:, 1],
            X[:, 2],
            color=color,
            label=label,
            alpha=alpha,
            linewidth=linewidth,
        )

    if waitTime is not None and waitTime != -1:
        plt.draw()
        plt.pause(waitTime)
    elif waitTime == None:
        plt.show(block=True)
    elif waitTime == -1:
        plt.show(block=False)


def plotPointSetsAsLine(
    ax: plt.axis,
    X: np.array,
    Y: np.array,
    labelX: str = None,
    labelY: str = None,
    colorX=[0, 0, 1],
    colorY=[1, 0, 0],
    linewidthX=1.5,
    linewidthY=1.5,
    alphaX=1,
    alphaY=1,
    waitTime=None,
):
    if labelX is None:
        ax.plot3D(
            X[:, 0], X[:, 1], X[:, 2], color=colorX, linewidth=linewidthX, alpha=alphaX
        )
    else:
        ax.plot3D(
            X[:, 0],
            X[:, 1],
            X[:, 2],
            color=colorX,
            label=labelX,
            linewidth=linewidthX,
            alpha=alphaX,
        )

    if labelY is None:
        ax.plot3D(
            Y[:, 0], Y[:, 1], Y[:, 2], color=colorY, linewidth=linewidthY, alpha=alphaY
        )
    else:
        ax.plot3D(
            Y[:, 0],
            Y[:, 1],
            Y[:, 2],
            color=colorY,
            label=labelY,
            linewidth=linewidthY,
            alpha=alphaY,
        )

    if waitTime is not None and waitTime != -1:
        plt.draw()
        plt.pause(waitTime)
    elif waitTime == None:
        plt.show(block=True)
    elif waitTime == -1:
        plt.show(block=False)


def plotPointSetAsColorGradedLine(
    ax: plt.axis,
    X: np.array,
    colorMap=plt.cm.gray,
    colorGrad=None,
    alpha=1,
    linewidth=1.5,
    waitTime=None,
):
    if colorGrad is None:
        colorGrad = np.linspace(0, 255, X[:, 0].size)

    for i in range(len(X[:, 0]) - 1):
        ax.plot3D(
            X[i : i + 2, 0],
            X[i : i + 2, 1],
            X[i : i + 2, 2],
            color=colorMap(colorGrad[i])[:3],
            alpha=alpha,
            linewidth=linewidth,
        )

    if waitTime is not None and waitTime != -1:
        plt.draw()
        plt.pause(waitTime)
    elif waitTime == None:
        plt.show(block=True)
    elif waitTime == -1:
        plt.show(block=False)
