import os
import sys
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

try:
    sys.path.append(os.getcwd().replace("/plot", ""))
    from src.modelling.topologyModel import topologyModel
    from src.simulation.bdloTemplates import initArenaWireHarness
    from src.visualization.plot3D import *
    from src.visualization.plot2D import *
except:
    print("Imports for plotting BFS example failed.")
    raise

saveFig = False
# saving paths
savePath = "/mnt/c/Users/ac129490/Documents/Dissertation/Thesis/62bebc3388a16f7dcc7f9153/figures/plots/"
fileName = "plotExampleBFS"
fileType = ".pdf"
# loading path
dataPath = "tests/testdata/topologyExtraction/wireHarnessReduced.txt"
# style settings
color_notExtracted = [0.7, 0.7, 0.7]
branchNodeSize = 50
leafNodeSize = 50

if __name__ == "__main__":
    testPointSet = np.loadtxt(dataPath)
    exampleTopology = initArenaWireHarness()
    localCoordinateSamples = np.linspace(0, 1, 10)
    # model = exampleTopology.getCartesianJointPositions()

    # flip smallest branch to match with figure from thesis# examlePositions = exampleTopology.getCartesianJointPositions()[:, -2::]
    exampleTopology.setBranchRootDof(5, 0, -np.pi / 4)

    pointPairs = exampleTopology.getAdjacentPointPairsAndBranchCorrespondance()

    # invert z and y and make everything 2D
    edges = []
    for pointPair in pointPairs:
        pointStart = np.array((pointPair[0][2], pointPair[0][1]))
        pointEnd = np.array((pointPair[1][2], pointPair[1][1]))
        edge = (np.stack((pointStart, pointEnd)), pointPair[2])
        edges.append(edge)

    # leafNode Coordiates
    leafNodeCoordinates = exampleTopology.getLeafNodeCartesianPositions()
    # invert z and y and make everything 2D
    leafNodeCoordinates[:, [1, 2]] = leafNodeCoordinates[:, [2, 1]]
    leafNodeCoordinates = leafNodeCoordinates[:, 1:]

    # branchNode Coordinates
    branchNodeCoordinates = exampleTopology.getBranchNodeCartesianPositions()
    # invert z and y and make everything 2D
    branchNodeCoordinates[:, [1, 2]] = branchNodeCoordinates[:, [2, 1]]
    branchNodeCoordinates = branchNodeCoordinates[:, 1:]
    branchNodeCoordinates = np.unique(
        np.round(branchNodeCoordinates, 2), axis=0
    )  # make unique coordinates

    # colormap
    colorMap = matplotlib.colormaps["Blues"]
    lowerLim = 0
    upperLim = np.round(exampleTopology.getNumBranches() - 1)
    norm = matplotlib.colors.Normalize(vmin=lowerLim, vmax=upperLim)  # Normalizer
    sm = plt.cm.ScalarMappable(cmap=colorMap, norm=norm)  # creating ScalarMappable

    # figure a) input
    fig_input, ax_input = setupLatexPlot2D(
        axisLimX=[-0.1, 1],
        axisLimY=[-0.3, 0.35],
    )
    for i, edge in enumerate(edges):
        plotColor = color_notExtracted
        plotLine(ax_input, pointPair=edge[0], color=plotColor)
        plotPoint(ax=ax_input, x=edge[0][0], color=plotColor)
        plotPoint(ax=ax_input, x=edge[0][1], color=plotColor)
    plt.grid(False)
    plt.axis("off")
    if saveFig:
        plt.savefig(
            savePath + fileName + "_" + "1" + fileType,
            bbox_inches="tight",
        )
    else:
        plt.show(block=False)

    # figure b) branch / leafnode extraction
    fig_branchLeafNodeExtraction, ax = setupLatexPlot2D(
        axisLimX=[-0.1, 1],
        axisLimY=[-0.3, 0.35],
    )
    for i, edge in enumerate(edges):
        plotColor = color_notExtracted
        plotLine(ax, pointPair=edge[0], color=plotColor)
        plotPoint(ax=ax, x=edge[0][0], color=plotColor)
        plotPoint(ax=ax, x=edge[0][1], color=plotColor)
    for leafNodeCoordinate in leafNodeCoordinates:
        plotPoint(
            ax=ax,
            x=leafNodeCoordinate,
            color=[1, 0, 0],
            size=leafNodeSize,
            marker="s",
            zorder=2,
        )
    for branchNodeCoordinate in branchNodeCoordinates:
        plotPoint(
            ax=ax,
            x=branchNodeCoordinate,
            color=[1, 0, 0],
            size=branchNodeSize,
            marker="^",
            zorder=2,
        )
    plt.grid(False)
    plt.axis("off")
    if saveFig:
        plt.savefig(
            savePath + fileName + "_" + "2" + fileType,
            bbox_inches="tight",
        )
    else:
        plt.show(block=False)

    # figure c) first branch node
    fig_firstBranchExtraction, ax = setupLatexPlot2D(
        axisLimX=[-0.1, 1],
        axisLimY=[-0.3, 0.35],
    )
    for i, edge in enumerate(edges):
        if (edge[1] == 0) or (edge[1] == 1) or (edge[1] == 2):
            plotColor = [1, 0, 0]
            plotLine(ax, pointPair=edge[0], color=plotColor)
            plotPoint(ax=ax, x=edge[0][0], color=plotColor)
            plotPoint(ax=ax, x=edge[0][1], color=plotColor)
        else:
            plotColor = color_notExtracted
            plotLine(ax, pointPair=edge[0], color=plotColor)
            plotPoint(ax=ax, x=edge[0][0], color=plotColor)
            plotPoint(ax=ax, x=edge[0][1], color=plotColor)
    for i, leafNodeCoordinate in enumerate(leafNodeCoordinates):
        plotColor = color_notExtracted
        plotPoint(
            ax=ax,
            x=leafNodeCoordinate,
            color=plotColor,
            size=leafNodeSize,
            marker="s",
            zorder=2,
        )
    for branchNodeCoordinate in branchNodeCoordinates:
        plotColor = color_notExtracted
        plotPoint(
            ax=ax,
            x=branchNodeCoordinate,
            color=plotColor,
            size=branchNodeSize,
            marker="^",
            zorder=2,
        )
    plotPoint(
        ax=ax,
        x=branchNodeCoordinates[0, :],
        color=[1, 0, 0],
        size=branchNodeSize,
        marker="^",
        zorder=3,
    )
    plt.grid(False)
    plt.axis("off")
    if saveFig:
        plt.savefig(
            savePath + fileName + "_" + "3" + fileType,
            bbox_inches="tight",
        )
    else:
        plt.show(block=False)

    # figure d) second branch node
    fig_firstBranchExtraction, ax = setupLatexPlot2D(
        axisLimX=[-0.1, 1],
        axisLimY=[-0.3, 0.35],
    )
    extractedBranches = [0, 1, 2]
    currentBranchNode = [1]
    visitedBranchNodes = [0]
    for i, edge in enumerate(edges):
        if (edge[1] == 3) or (edge[1] == 4):
            plotColor = [1, 0, 0]
        elif edge[1] in extractedBranches:
            plotColor = [
                sm.to_rgba(edge[1])[0],
                sm.to_rgba(edge[1])[1],
                sm.to_rgba(edge[1])[2],
            ]
        else:
            plotColor = color_notExtracted
        plotLine(ax, pointPair=edge[0], color=plotColor)
        plotPoint(ax=ax, x=edge[0][0], color=plotColor)
        plotPoint(ax=ax, x=edge[0][1], color=plotColor)

    leafNodeCounter = 0
    for i, branch in enumerate(exampleTopology.branches):
        if i in extractedBranches:
            plotColor = [
                sm.to_rgba(i)[0],
                sm.to_rgba(i)[1],
                sm.to_rgba(i)[2],
            ]
        else:
            plotColor = color_notExtracted
        if exampleTopology.getNumLeafNodesFromBranch(branch) == 1:
            plotPoint(
                ax=ax,
                x=leafNodeCoordinates[leafNodeCounter],
                color=plotColor,
                size=leafNodeSize,
                marker="s",
                zorder=2,
            )
            leafNodeCounter += 1
    for i, branchNodeCoordinate in enumerate(branchNodeCoordinates):
        if i in visitedBranchNodes:
            plotColor = [
                sm.to_rgba(0)[0],
                sm.to_rgba(0)[1],
                sm.to_rgba(0)[2],
            ]
        elif i in currentBranchNode:
            plotColor = [1, 0, 0]
        else:
            plotColor = color_notExtracted
        plotPoint(
            ax=ax,
            x=branchNodeCoordinate,
            color=plotColor,
            size=branchNodeSize,
            marker="^",
            zorder=2,
        )
    plt.grid(False)
    plt.axis("off")
    if saveFig:
        plt.savefig(
            savePath + fileName + "_" + "4" + fileType,
            bbox_inches="tight",
        )
    else:
        plt.show(block=False)

    # figure e) thirdbranch node
    fig_firstBranchExtraction, ax = setupLatexPlot2D(
        axisLimX=[-0.1, 1],
        axisLimY=[-0.3, 0.35],
    )
    extractedBranches = [0, 1, 2, 3, 4]
    currentBranches = [5, 6]
    currentBranchNode = [2]
    visitedBranchNodes = [0, 1]
    for i, edge in enumerate(edges):
        if edge[1] in currentBranches:
            plotColor = [1, 0, 0]
        elif edge[1] in extractedBranches:
            plotColor = [
                sm.to_rgba(edge[1])[0],
                sm.to_rgba(edge[1])[1],
                sm.to_rgba(edge[1])[2],
            ]
        else:
            plotColor = color_notExtracted
        plotLine(ax, pointPair=edge[0], color=plotColor)
        plotPoint(ax=ax, x=edge[0][0], color=plotColor)
        plotPoint(ax=ax, x=edge[0][1], color=plotColor)

    leafNodeCounter = 0
    for i, branch in enumerate(exampleTopology.branches):
        if i in extractedBranches:
            plotColor = [
                sm.to_rgba(i)[0],
                sm.to_rgba(i)[1],
                sm.to_rgba(i)[2],
            ]
        else:
            plotColor = color_notExtracted
        if exampleTopology.getNumLeafNodesFromBranch(branch) == 1:
            plotPoint(
                ax=ax,
                x=leafNodeCoordinates[leafNodeCounter],
                color=plotColor,
                size=leafNodeSize,
                marker="s",
                zorder=2,
            )
            leafNodeCounter += 1
    for i, branchNodeCoordinate in enumerate(branchNodeCoordinates):
        if i in visitedBranchNodes:
            plotColor = [
                sm.to_rgba(0)[0],
                sm.to_rgba(0)[1],
                sm.to_rgba(0)[2],
            ]
        elif i in currentBranchNode:
            plotColor = [1, 0, 0]
        else:
            plotColor = color_notExtracted
        plotPoint(
            ax=ax,
            x=branchNodeCoordinate,
            color=plotColor,
            size=branchNodeSize,
            marker="^",
            zorder=2,
        )
    plt.grid(False)
    plt.axis("off")
    if saveFig:
        plt.savefig(
            savePath + fileName + "_" + "5" + fileType,
            bbox_inches="tight",
        )
    else:
        plt.show(block=False)

    # figure f) extracted topology model
    fig_firstBranchExtraction, ax = setupLatexPlot2D(
        axisLimX=[-0.1, 1],
        axisLimY=[-0.3, 0.35],
    )
    extractedBranches = [0, 1, 2, 3, 4, 5, 6]
    currentBranches = []
    currentBranchNode = []
    visitedBranchNodes = [0, 1, 2]
    for i, edge in enumerate(edges):
        plotColor = [
            sm.to_rgba(edge[1])[0],
            sm.to_rgba(edge[1])[1],
            sm.to_rgba(edge[1])[2],
        ]
        plotLine(ax, pointPair=edge[0], color=plotColor)
        plotPoint(ax=ax, x=edge[0][0], color=plotColor)
        plotPoint(ax=ax, x=edge[0][1], color=plotColor)

    leafNodeCounter = 0
    for i, branch in enumerate(exampleTopology.branches):
        if i in extractedBranches:
            plotColor = [
                sm.to_rgba(i)[0],
                sm.to_rgba(i)[1],
                sm.to_rgba(i)[2],
            ]
        else:
            plotColor = color_notExtracted
        if exampleTopology.getNumLeafNodesFromBranch(branch) == 1:
            plotPoint(
                ax=ax,
                x=leafNodeCoordinates[leafNodeCounter],
                color=plotColor,
                size=leafNodeSize,
                marker="s",
                zorder=2,
            )
            leafNodeCounter += 1
    for i, branchNodeCoordinate in enumerate(branchNodeCoordinates):
        if i in visitedBranchNodes:
            plotColor = [
                sm.to_rgba(0)[0],
                sm.to_rgba(0)[1],
                sm.to_rgba(0)[2],
            ]
        elif i in currentBranchNode:
            plotColor = [1, 0, 0]
        else:
            plotColor = color_notExtracted
        plotPoint(
            ax=ax,
            x=branchNodeCoordinate,
            color=plotColor,
            size=branchNodeSize,
            marker="^",
            zorder=2,
        )
    plt.grid(False)
    plt.axis("off")
    if saveFig:
        plt.savefig(
            savePath + fileName + "_" + "6" + fileType,
            bbox_inches="tight",
        )
    else:
        plt.show(block=True)
