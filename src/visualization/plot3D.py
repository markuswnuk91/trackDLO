import sys, os
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from warnings import warn

try:
    sys.path.append(os.getcwd().replace("/src/visualization", ""))
    from src.visualization.plotUtils import *
    from src.visualization.colors import *
    from src.modelling.topologyModel import topologyModel
except:
    print("Imports for Plot3D File failed.")
    raise


def print_axis_view_settings(ax):
    print("axis limits:\n")
    print(ax.get_xlim())
    print(ax.get_ylim())
    print(ax.get_zlim())

    print("azimut and elevation:\n")
    print(ax.azim)
    print(ax.elev)


def setupLatexPlot3D(
    figureWidth=483.6969,
    figureHeight=None,
    widthHeightRatio=None,
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
    widthHeightRatio = 1 if widthHeightRatio is None else widthHeightRatio
    if figureHeight is not None:
        width, height = set_size(width=figureWidth, height=figureHeight)
    else:
        width, height = set_size(width=figureWidth)
    fig = plt.figure(figsize=(width, height))
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


def plotSinglePoint(
    ax,
    x,
    color=[0, 0, 1],
    edgeColor=None,
    alpha=1,
    label: str = None,
    size=None,
    marker=None,
    zOrder=None,
):
    zOrder = 1 if zOrder is None else zOrder
    edgeColor = color if edgeColor is None else edgeColor
    size = 20 if size is None else size
    marker = "o" if marker is None else marker
    ax.plot(
        x[0],
        x[1],
        x[2],
        marker=marker,
        zorder=zOrder,
        color=color,
        markersize=size,
        markeredgecolor=edgeColor,
    )


def plotPoint(
    ax,
    x,
    color=[0, 0, 1],
    edgeColor=None,
    alpha=1,
    label: str = None,
    size=None,
    marker="o",
    zOrder=None,
):
    zOrder = 1 if zOrder is None else zOrder
    edgeColor = color if edgeColor is None else edgeColor
    size = 20 if size is None else size
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
                zorder=zOrder,
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
                zorder=zOrder,
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
                zorder=zOrder,
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
                zorder=zOrder,
            )


def plotPointSet(
    ax,
    X,
    color=[0, 0, 1],
    edgeColor=None,
    size=None,
    markerStyle=None,
    lineWidth=None,
    alpha=None,
    label: str = None,
    zOrder=None,
):
    size = 20 if size is None else size
    markerStyle = "o" if markerStyle is None else markerStyle
    alpha = 1 if alpha is None else alpha
    edgeColor = color if edgeColor is None else edgeColor
    zOrder = 1 if zOrder is None else zOrder
    lineWidth = 1.5 if lineWidth is None else lineWidth
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
                linewidth=lineWidth,
                zorder=zOrder,
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
                zorder=zOrder,
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
                zorder=zOrder,
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
                zorder=zOrder,
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


def plotPointCloud(
    ax,
    points,
    colors,
    size=None,
    markerStyle=None,
    alpha=None,
):
    markerStyle = "o" if markerStyle is None else markerStyle
    size = 5 if size is None else size
    alpha = 1 if alpha is None else alpha
    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=colors,
        s=size,
        alpha=alpha,
        marker=markerStyle,
    )
    return ax


def plotLine(
    ax: plt.axis,
    pointPair: np.array,
    label: str = None,
    color=None,
    alpha=None,
    linewidth=None,
    lineStyle=None,
    zOrder=None,
):
    color = [0, 0, 1] if color is None else color
    alpha = 1 if alpha is None else alpha
    linewidth = 1.5 if linewidth is None else linewidth
    lineStyle = "-" if lineStyle is None else lineStyle
    zOrder = 1 if zOrder is None else zOrder

    if pointPair.shape[1] == 2:
        ax.plot(
            pointPair[:, 0],
            pointPair[:, 1],
            color=color,
            label=label,
            alpha=alpha,
            linewidth=linewidth,
            linestyle=lineStyle,
            zorder=zOrder,
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
            linestyle=lineStyle,
            zorder=zOrder,
        )


def plotPointSetAsLine(
    ax: plt.axis,
    X: np.array,
    label: str = None,
    color=[0, 0, 1],
    alpha=1,
    linewidth=1.5,
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
    ax.plot_wireframe(xx, yy, z_range[0] * np.ones(4).reshape(2, 2), color=color)
    ax.plot_surface(
        xx, yy, z_range[0] * np.ones(4).reshape(2, 2), color=color, alpha=alpha
    )
    ax.plot_wireframe(xx, yy, z_range[1] * np.ones(4).reshape(2, 2), color=color)
    ax.plot_surface(
        xx, yy, z_range[1] * np.ones(4).reshape(2, 2), color=color, alpha=alpha
    )

    yy, zz = np.meshgrid(y_range, z_range)
    ax.plot_wireframe(x_range[0] * np.ones(4).reshape(2, 2), yy, zz, color=color)
    ax.plot_surface(
        x_range[0] * np.ones(4).reshape(2, 2), yy, zz, color=color, alpha=alpha
    )
    ax.plot_wireframe(x_range[1] * np.ones(4).reshape(2, 2), yy, zz, color=color)
    ax.plot_surface(
        x_range[1] * np.ones(4).reshape(2, 2), yy, zz, color=color, alpha=alpha
    )

    xx, zz = np.meshgrid(x_range, z_range)
    ax.plot_wireframe(xx, y_range[0] * np.ones(4).reshape(2, 2), zz, color=color)
    ax.plot_surface(
        xx, y_range[0] * np.ones(4).reshape(2, 2), zz, color=color, alpha=alpha
    )
    ax.plot_wireframe(xx, y_range[1] * np.ones(4).reshape(2, 2), zz, color=color)
    ax.plot_surface(
        xx, y_range[1] * np.ones(4).reshape(2, 2), zz, color=color, alpha=alpha
    )


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


def plotArrow3D(
    ax,
    startPoint,
    endPoint,
    size=None,
    color=None,
    alpha=None,
    fillColor=None,
    arrowStyle=None,
    lineStyle=None,
):
    color = [0, 0, 0] if color is None else color
    alpha = 1 if alpha is None else alpha
    size = 20 if size is None else size
    arrowStyle = "-|>" if arrowStyle is None else arrowStyle
    lineStyle = "dashed" if lineStyle is None else lineStyle
    fillColor = None if fillColor is None else fillColor
    drawArrow3D(
        ax,
        startPoint[0],
        startPoint[1],
        startPoint[2],
        endPoint[0] - startPoint[0],
        endPoint[1] - startPoint[1],
        endPoint[2] - startPoint[2],
        mutation_scale=size,
        arrowstyle=arrowStyle,
        linestyle=lineStyle,
        ec=color,
        fc=fillColor,
        alpha=alpha,
    )


def plotArrows3D(
    ax,
    startPoints,
    endPoints,
    size=None,
    color=None,
    alpha=None,
    fillColor=None,
    arrowStyle=None,
    lineStyle=None,
):
    color = [0, 0, 0] if color is None else color
    alpha = 1 if alpha is None else alpha
    size = 20 if size is None else size
    arrowStyle = "-|>" if arrowStyle is None else arrowStyle
    lineStyle = "-" if lineStyle is None else lineStyle
    fillColor = None if fillColor is None else fillColor
    for x_start, x_end in zip(startPoints, endPoints):
        # if np.linalg.norm(x_old - x_new) > 0.05:
        plotArrow3D(
            ax=ax,
            startPoint=x_start,
            endPoint=x_start,
            size=size,
            color=color,
            alpha=alpha,
            fillColor=fillColor,
            arrowStyle=arrowStyle,
            lineStyle=lineStyle,
        )


def plotGraph3D(
    ax,
    X,
    adjacencyMatrix,
    pointColor=None,
    lineColor=None,
    pointSize=None,
    lineWidth=None,
    lineStyle=None,
    pointAlpha=None,
    lineAlpha=None,
    zOrder=None,
):
    pointColor = [0, 0, 1] if pointColor is None else pointColor
    lineColor = [0, 0, 1] if lineColor is None else lineColor
    pointSize = 10 if pointSize is None else pointSize
    lineWidth = 1.5 if lineWidth is None else lineWidth
    pointAlpha = 1 if pointAlpha is None else pointAlpha
    lineAlpha = 1 if lineAlpha is None else lineAlpha
    lineStyle = "-" if lineStyle is None else lineStyle
    zOrder = 1 if zOrder is None else zOrder
    plotPointSet(
        ax=ax, X=X, color=pointColor, size=pointSize, alpha=pointAlpha, zOrder=zOrder
    )
    # check if matrix is symmetric
    if not np.allclose(adjacencyMatrix, adjacencyMatrix.T, rtol=1e-05, atol=1e-08):
        warn("Provided adjacency matrix is not symmetric!")
    I, J = adjacencyMatrix.shape
    for i in range(0, I):
        for j in range(i, J):
            if adjacencyMatrix[i, j] != 0:
                plotLine(
                    ax=ax,
                    pointPair=np.vstack((X[i, :], X[j, :])),
                    color=lineColor,
                    linewidth=lineWidth,
                    alpha=lineAlpha,
                    lineStyle=lineStyle,
                    zOrder=zOrder,
                )
    return ax


# TODO: implement this fuction for tracking Evaluation
def plotBranchWiseColoredGraph3D(
    ax,
    positions3D,
    adjacencyMatrix,
    branchCorrespondanceMatrix,
    colorPalette=None,
    lineWidth=None,
    pointSize=None,
):
    colorPalette = (
        thesisColorPalettes["viridis"] if colorPalette is None else colorPalette
    )

    numBranches = branchCorrespondanceMatrix.shape[1]

    colorScaleCoordinates = np.linspace(0, 1, numBranches)
    branchColors = []
    for s in colorScaleCoordinates:
        branchColors.append(colorPalette.to_rgba(s)[:3])

    for branchIndex in range(0, numBranches):
        indices = np.where(branchCorrespondanceMatrix[:, branchIndex] == 1)[0]
        branchPositions = positions3D[indices, :]
        branchAdjacencyMatrix = np.array(
            [[adjacencyMatrix[row][col] for col in indices] for row in indices]
        )
        ax = plotGraph3D(
            ax=ax,
            X=branchPositions,
            adjacencyMatrix=branchAdjacencyMatrix,
            pointColor=branchColors[branchIndex],
            lineColor=branchColors[branchIndex],
            lineWidth=lineWidth,
            pointSize=pointSize,
        )
        # rgbImg = plotGraph2_CV(
        #     rgbImg=rgbImg,
        #     positions2D=branchPositions,
        #     adjacencyMatrix=branchAdjacencyMatrix,
        #     lineColor=branchColors[branchIndex],
        #     circleColor=branchColors[branchIndex],
        #     lineThickness=lineThickness,
        #     circleRadius=circleRadius,
        # )
    return ax


# def plotBranchColoredGraph3D(
#     ax,
#     X,
#     adjacencyMatrix,
#     colorPalette=None
#     pointColor=None,
#     lineColor=None,
#     pointSize=None,
#     lineWidth=None,
#     pointAlpha=None,
#     lineAlpha=None,
#     zOrder=None,
# ):
#     colorPalette = thesisColorPalettes["viridis"] if colorPalette is None else colorPalette
#     pointColor = [0, 0, 1] if pointColor is None else pointColor
#     lineColor = [0, 0, 1] if lineColor is None else lineColor
#     pointSize = 10 if pointSize is None else pointSize
#     lineWidth = 1.5 if lineWidth is None else lineWidth
#     pointAlpha = 1 if pointAlpha is None else pointAlpha
#     lineAlpha = 1 if lineAlpha is None else lineAlpha
#     zOrder = 1 if zOrder is None else zOrder

#     topology = topologyModel(adjacencyMatrix=adjacencyMatrix)
#     topology.getBranchIndices()
#     plotPointSet(
#         ax=ax, X=X, color=pointColor, size=pointSize, alpha=pointAlpha, zOrder=zOrder
#     )
#     i = 0
#     j = 0
#     I, J = adjacencyMatrix.shape
#     for i in range(0, I):
#         for j in range(0, J):
#             if adjacencyMatrix[i, j] != 0:
#                 plotLine(
#                     ax=ax,
#                     pointPair=np.vstack((X[i, :], X[j, :])),
#                     color=lineColor,
#                     linewidth=lineWidth,
#                     alpha=lineAlpha,
#                     zOrder=zOrder,
#                 )
#                 plotPoint(ax=ax,color=)
#     return


def plotTopology3D(
    ax,
    topology,
    color=None,
    lineWidth=None,
    lineAlpha=None,
    plotPoints=False,
    pointAlpha=None,
    pointSize=None,
):
    color = [0, 0, 1] if color is None else color

    pointPairs = topology.getAdjacentPointPairs()
    for pointPair in pointPairs:
        stackedPair = np.stack(pointPair)
        plotLine(
            ax,
            pointPair=stackedPair,
            color=color,
            linewidth=lineWidth,
            alpha=lineAlpha,
        )
    if plotPoints:
        points = np.unique(
            np.array([point for pointPair in pointPairs for point in pointPair]), axis=0
        )
        plotPointSet(ax, X=points, color=color, size=pointSize, alpha=pointAlpha)
    return


def plotBranchWiseColoredTopology3D(
    ax,
    topology,
    colorPalette=None,
    lineWidth=None,
    pointSize=None,
    lineAlpha=None,
    plotPoints=True,
    pointAlpha=None,
    zOrder=None,
):
    if colorPalette is None:
        colorPalette = thesisColorPalettes["viridis"]
    zOrder = zOrder if zOrder is None else zOrder
    pointAlpha = 1 if pointAlpha is None else pointAlpha
    connections = topology.getAdjacentPointPairsAndBranchCorrespondance()
    numBranches = topology.getNumBranches()
    colorScaleCoordinates = np.linspace(0, 1, numBranches)
    branchColors = []
    for s in colorScaleCoordinates:
        branchColors.append(colorPalette.to_rgba(s)[:3])
    for connection in connections:
        stackedPair = np.stack(connection[:2])
        branchIndex = connection[2]
        # plotColor = [
        #     sm.to_rgba(branchNumber)[0],
        #     sm.to_rgba(branchNumber)[1],
        #     sm.to_rgba(branchNumber)[2],
        # ]
        plotLine(
            ax=ax,
            pointPair=stackedPair,
            color=branchColors[branchIndex],
            linewidth=lineWidth,
            alpha=lineAlpha,
            zOrder=zOrder,
        )
        if plotPoints:
            # plotPoint(
            #     ax=ax,
            #     x=stackedPair[0, :],
            #     color=branchColors[branchIndex],
            #     size=pointSize,
            #     zOrder=zOrder,
            # )
            # plotPoint(
            #     ax=ax,
            #     x=stackedPair[1, :],
            #     color=branchColors[branchIndex],
            #     size=pointSize,
            #     zOrder=zOrder,
            # )
            ax.plot(
                [stackedPair[0, 0]],
                [stackedPair[0, 1]],
                [stackedPair[0, 2]],
                marker="o",
                alpha=pointAlpha,
                zorder=zOrder,
                color=branchColors[branchIndex],
                markersize=pointSize,
            )
            ax.plot(
                [stackedPair[1, 0]],
                [stackedPair[1, 1]],
                [stackedPair[1, 2]],
                marker="o",
                alpha=pointAlpha,
                zorder=zOrder,
                color=branchColors[branchIndex],
                markersize=pointSize,
            )

    return ax


def plotCorrespondances3D(
    ax,
    X,
    Y,
    C=None,
    xColor=None,
    yColor=None,
    correspondanceColor=None,
    xSize=None,
    ySize=None,
    linewidth=None,
    xAlpha=None,
    yAlpha=None,
    lineAlpha=None,
):
    (N, D) = X.shape
    C = np.eye(N, N) if C is None else C
    xColor = [0, 0, 1] if xColor is None else xColor
    yColor = [1, 0, 0] if yColor is None else yColor
    correspondanceColor = (
        [0.3, 0.3, 0.3] if correspondanceColor is None else correspondanceColor
    )

    XCorresponding = C @ X
    for x, y in zip(XCorresponding, Y):
        plotPoint(ax=ax, x=x, color=xColor, alpha=xAlpha, size=xSize)
        plotPoint(ax=ax, x=y, color=yColor, alpha=yAlpha, size=ySize)
        plotLine(
            ax=ax,
            pointPair=np.vstack((x, y)),
            color=correspondanceColor,
            linewidth=linewidth,
            alpha=lineAlpha,
        )
    return


def plotCorrespondancesAsArrows3D(
    ax,
    X,
    Y,
    C=None,
    xColor=None,
    yColor=None,
    correspondanceColor=None,
    xSize=None,
    ySize=None,
    linestyle=None,
    linewidth=None,
    headSize=None,
    xAlpha=None,
    yAlpha=None,
    lineAlpha=None,
):
    (N, D) = X.shape
    C = np.eye(N, N) if C is None else C
    xColor = [0, 0, 1] if xColor is None else xColor
    yColor = [1, 0, 0] if yColor is None else yColor
    correspondanceColor = (
        [0.3, 0.3, 0.3] if correspondanceColor is None else correspondanceColor
    )
    linestyle = "-" if linestyle is None else linestyle
    headSize = 20 if headSize is None else headSize
    XCorresponding = C @ X
    for x, y in zip(XCorresponding, Y):
        plotPoint(ax=ax, x=x, color=xColor, alpha=xAlpha, size=xSize)
        plotPoint(ax=ax, x=y, color=yColor, alpha=yAlpha, size=ySize)
        drawArrow3D(
            ax,
            x[0],
            x[1],
            x[2],
            y[0] - x[0],
            y[1] - x[1],
            y[2] - x[2],
            mutation_scale=20,
            arrowstyle="-|>",
            linestyle=linestyle,
        )
    return
