import sys
import os
import matplotlib.pyplot as plt

try:
    sys.path.append(os.getcwd().replace("/plot", ""))
    from src.sensing.dataHandler import DataHandler
    from src.simulation.bdlo import BranchedDeformableLinearObject
    from src.visualization.plot3D import *
    from src.visualization.colors import *
    from src.visualization.plotUtils import scale_axes_to_fit
except:
    print("Imports for plotting tolology extraction failed.")
    raise

relFilePath = "data/darus_data_download/data/20230516_Configurations_labeled/20230516_115857_arena/data/20230516_120332_090647_image_rgb.png"


def plot_feature_branch_length(bdloModel):
    fig, ax = setupLatexPlot3D()
    X, A = bdloModel.getJointPositionsAndAdjacencyMatrix()
    plotGraph3D(
        ax=ax, X=X, adjacencyMatrix=A, pointColor=[0, 0, 1], lineColor=[0, 0, 1]
    )
    scale_axes_to_fit(ax=ax, points=X)
    # color branch for which the features are displayed
    branchIndex = 1
    correspondingBodyNodes = bdloModel.getBranch(branchIndex).getBranchInfo()[
        "correspondingBodyNodeIndices"
    ]
    correspondingBodyNodes.append(correspondingBodyNodes[-1] + 1)
    X = bdloModel.getCartesianJointPositions()
    plotPointSet(ax=ax, X=X[correspondingBodyNodes, :], color=[1, 0, 0])
    plotPointSetAsLine(ax=ax, X=X[correspondingBodyNodes, :], color=[1, 0, 0])
    plt.axis("off")


def plot_feature_num_leafnodes(bdloModel):
    fig, ax = setupLatexPlot3D()
    X, A = bdloModel.getJointPositionsAndAdjacencyMatrix()
    plotGraph3D(
        ax=ax, X=X, adjacencyMatrix=A, pointColor=[0, 0, 1], lineColor=[0, 0, 1]
    )
    scale_axes_to_fit(ax=ax, points=X)
    # color leaf node
    branchIndex = 1
    if bdloModel.getNumLeafNodesFromBranch(bdloModel.getBranch(branchIndex)) == 1:
        leafNodeCoordinates = bdloModel.getCartesianPositionFromBranchLocalCoordinate(
            branchIndex, 1
        )
        plotPoint(ax=ax, x=leafNodeCoordinates, color=[1, 0, 0])
    plt.axis("off")


if __name__ == "__main__":
    # configure file paths
    fileName = os.path.basename(relFilePath)
    dataFolderPath = os.path.dirname(relFilePath)
    dataSetPath = os.path.dirname(dataFolderPath) + "/"

    # setup hepler classes
    dataHandler = DataHandler()

    # load model
    numBodyNodes = 30
    modelInfo = dataHandler.loadModelParameters("model.json", dataSetPath)
    bdloModel = BranchedDeformableLinearObject(
        **{
            "adjacencyMatrix": modelInfo["topologyModel"],
            "branchSpecs": list(modelInfo["branchSpecifications"].values()),
            "defaultNumBodyNodes": numBodyNodes,
        }
    )

    # plot_feature_branch_length(bdloModel)
    plot_feature_num_leafnodes(bdloModel)
    plt.show(block=True)
    print("Done.")
