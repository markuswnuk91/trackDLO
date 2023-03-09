import matplotlib.pyplot as plt
import numpy as np


def set_axes_equal3D(ax):
    """Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


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


def set_axes_equal(ax):
    if ax.name == "3d":
        set_axes_equal3D(ax)
    else:
        ax.set_aspect("equal", adjustable="box")


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
    ax,
    x,
    color=[0, 0, 1],
    edgeColor=None,
    alpha=1,
    label: str = None,
    size=20,
):
    edgeColor = color if edgeColor is None else edgeColor
    if len(x) == 3:
        if label is None:
            ax.scatter(
                x[0],
                x[1],
                x[2],
                color=color,
                alpha=alpha,
                s=size,
                edgecolors=edgeColor,
            )
        else:
            ax.scatter(
                x[0],
                x[1],
                x[2],
                color=color,
                label=label,
                alpha=alpha,
                s=size,
                edgecolors=edgeColor,
            )
    elif len(x) == 2:
        if label is None:
            ax.scatter(
                x[0], x[1], color=color, alpha=alpha, s=size, edgecolors=edgeColor
            )
        else:
            ax.scatter(
                x[0],
                x[1],
                color=color,
                label=label,
                alpha=alpha,
                s=size,
                edgecolors=edgeColor,
            )


def plotPointSet(
    ax,
    X,
    color=[0, 0, 1],
    edgeColor=None,
    size=None,
    markerStyle=None,
    alpha=None,
    label: str = None,
    axisEqual=None,
):
    size = 20 if size is None else size
    markerStyle = "o" if markerStyle is None else markerStyle
    alpha = 1 if alpha is None else alpha
    edgeColor = color if edgeColor is None else edgeColor

    if X.shape[1] == 3:
        if label is None:
            ax.scatter(
                X[:, 0],
                X[:, 1],
                X[:, 2],
                color=color,
                alpha=alpha,
                s=size,
                marker=markerStyle,
                edgecolors=edgeColor,
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

    elif X.shape[1] == 2:
        if label is None:
            ax.scatter(
                X[:, 0],
                X[:, 1],
                color=color,
                alpha=alpha,
                s=size,
                marker=markerStyle,
                edgecolors=edgeColor,
            )
        else:
            ax.scatter(
                X[:, 0],
                X[:, 1],
                color=color,
                label=label,
                alpha=alpha,
                s=size,
                marker=markerStyle,
                edgecolor=None,
            )
    else:
        raise NotImplementedError


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
):

    xSize = 20 if xSize is None else xSize
    ySize = 20 if xSize is None else xSize
    xMarkerStyle = "o" if xMarkerStyle is None else xMarkerStyle
    yMarkerStyle = "o" if yMarkerStyle is None else yMarkerStyle
    xAlpha = 1 if xAlpha is None else xAlpha
    yAlpha = 1 if yAlpha is None else yAlpha

    if xLabel is not None:
        plotPointSet(
            ax=ax,
            X=X,
            color=xColor,
            edgeColor=xEdgeColor,
            size=xSize,
            markerStyle=xMarkerStyle,
            alpha=xAlpha,
            label=xLabel,
        )
    else:
        plotPointSet(
            ax=ax,
            X=X,
            color=xColor,
            edgeColor=xEdgeColor,
            size=xSize,
            markerStyle=xMarkerStyle,
            alpha=xAlpha,
        )
    if yLabel is not None:
        plotPointSet(
            ax=ax,
            X=Y,
            color=yColor,
            edgeColor=yEdgeColor,
            size=ySize,
            markerStyle=yMarkerStyle,
            alpha=yAlpha,
            label=yLabel,
        )
    else:
        plotPointSet(
            ax=ax,
            X=Y,
            color=yColor,
            edgeColor=yEdgeColor,
            size=ySize,
            markerStyle=yMarkerStyle,
            alpha=yAlpha,
        )


def plotLine(
    ax: plt.axis,
    pointPair: np.array,
    label: str = None,
    color=[0, 0, 1],
    alpha=1,
    linewidth=1.5,
):
    if pointPair.shape[1] == 2:
        ax.plot(
            pointPair[:, 0],
            pointPair[:, 1],
            color=color,
            label=label,
            alpha=alpha,
            linewidth=linewidth,
        )
    elif pointPair.shape[1] == 3:
        ax.plot3D(
            pointPair[:, 0],
            pointPair[:, 1],
            pointPair[:, 2],
            color=color,
            label=label,
            alpha=alpha,
            linewidth=linewidth,
        )


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
