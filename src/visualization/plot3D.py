import sys, os
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

try:
    sys.path.append(os.getcwd().replace("/src/visualization", ""))
    from src.visualization.plotUtils import set_size, set_axes_equal
except:
    print("Imports for Plot3D File failed.")
    raise


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
    marker="o",
    zorder=1,
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
                marker=marker,
                zorder=zorder,
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
                marker=marker,
                zorder=zorder,
            )
    elif len(x) == 2:
        if label is None:
            ax.scatter(
                x[0],
                x[1],
                color=color,
                alpha=alpha,
                s=size,
                edgecolors=edgeColor,
                marker=marker,
                zorder=zorder,
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
                marker=marker,
                zorder=zorder,
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
                ccolor=color,
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
    ySize = 20 if ySize is None else ySize
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


def plotCube(ax, x_Min, x_Max, y_Min, y_Max, z_Min, z_Max, color="r", alpha=0.2):
    x_range = np.array([x_Min, x_Max])
    y_range = np.array([y_Min, y_Max])
    z_range = np.array([z_Min, z_Max])

    xx, yy = np.meshgrid(x_range, y_range)
    ax.plot_wireframe(xx, yy, z_range[0] * np.ones(4).reshape(2, 2), color="r")
    ax.plot_surface(xx, yy, z_range[0] * np.ones(4).reshape(2, 2), color="r", alpha=0.2)
    ax.plot_wireframe(xx, yy, z_range[1] * np.ones(4).reshape(2, 2), color="r")
    ax.plot_surface(xx, yy, z_range[1] * np.ones(4).reshape(2, 2), color="r", alpha=0.2)

    yy, zz = np.meshgrid(y_range, z_range)
    ax.plot_wireframe(x_range[0] * np.ones(4).reshape(2, 2), yy, zz, color="r")
    ax.plot_surface(x_range[0] * np.ones(4).reshape(2, 2), yy, zz, color="r", alpha=0.2)
    ax.plot_wireframe(x_range[1] * np.ones(4).reshape(2, 2), yy, zz, color="r")
    ax.plot_surface(x_range[1] * np.ones(4).reshape(2, 2), yy, zz, color="r", alpha=0.2)

    xx, zz = np.meshgrid(x_range, z_range)
    ax.plot_wireframe(xx, y_range[0] * np.ones(4).reshape(2, 2), zz, color="r")
    ax.plot_surface(xx, y_range[0] * np.ones(4).reshape(2, 2), zz, color="r", alpha=0.2)
    ax.plot_wireframe(xx, y_range[1] * np.ones(4).reshape(2, 2), zz, color="r")
    ax.plot_surface(xx, y_range[1] * np.ones(4).reshape(2, 2), zz, color="r", alpha=0.2)


def plotVector(ax, origin, direction, length=None, color=None):
    length = np.linalg.norm(direction) if length is None else length
    color = [0, 0, 1] if color is None else color

    direction = length / np.linalg.norm(direction) * direction
    ax.quiver(
        origin[0],
        origin[1],
        origin[2],
        direction[0],
        direction[1],
        direction[2],
        color=color,
    )
