import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.csgraph import johnson

try:
    sys.path.append(os.getcwd().replace("/eval", ""))
    from src.evaluation.initialLocalization.initialLocalizationEvaluation import (
        InitialLocalizationEvaluation,
    )

    # visualization
    from src.visualization.plot3D import *

    from src.utils.utils import minimalSpanningTree
    from scipy.spatial import distance_matrix
except:
    print("Imports for initialization evaluation script failed.")
    raise

runOpt = {
    "dataSetsToEvaluate": [1],
    "runInitializationExperiment": True,
    "dataSetsToEvaluate" "runSom": False,
    "runL1": False,
    #    "evaluation": True
}

framesToEvaluate = [3]

dataSetPaths = [
    "data/darus_data_download/data/202230603_Configurations_mounted/20230603_143937_modelY/",
    "data/darus_data_download/data/20230807_Configurations_mounted/20230807_150735_partial/",
    "data/darus_data_download/data/202230603_Configurations_mounted/20230603_140143_arena/",
]


def find_n_longest_paths(min_span_tree, n_longest_paths):
    longest_paths = []
    current_graph = min_span_tree
    for n in range(0, n_longest_paths):
        dist_matrix, predecessors = johnson(
            csgraph=current_graph, directed=False, return_predecessors=True
        )
        dist_matrix[np.isinf(dist_matrix)] = 0
        max_distant_point_idxs = np.unravel_index(
            np.argmax(dist_matrix, axis=None), dist_matrix.shape
        )
        # remove the found longest path from the graph
        path = [max_distant_point_idxs[1]]
        predecessor = path[0]
        while predecessor != max_distant_point_idxs[0]:
            predecessor = predecessors[max_distant_point_idxs[0], predecessor]
            path.append(predecessor)
        # for row_idx in path:
        #     for col_idx in path:
        #         current_graph[row_idx, col_idx] = 0
        predecessorIdxs = path[1:]
        nodeIdxs = path[:-1]
        for nodeIdx, predecessorIdx in zip(nodeIdxs, predecessorIdxs):
            current_graph[nodeIdx, predecessorIdx] = 0
            current_graph[predecessorIdx, nodeIdx] = 0
        longest_paths.append(path)
    return longest_paths


if __name__ == "__main__":
    if runOpt["dataSetsToEvaluate"][0] == -1:
        dataSetsToEvaluate = dataSetPaths
    else:
        dataSetsToEvaluate = [
            dataSetPath
            for i, dataSetPath in enumerate(dataSetPaths)
            if i in runOpt["dataSetsToEvaluate"]
        ]

    # determine data sets without too much occlusion/noise

    # run initialization experiments
    if runOpt["runInitializationExperiment"]:
        for dataSetPath in dataSetsToEvaluate:
            # setup evaluation class
            dataSetName = dataSetPath.split("/")[-2]
            modelIdentifier = dataSetName.split("_")[-1]
            relConfigFilePath = configFilePath = (
                "/evalConfigs/evalConfig" + "_" + modelIdentifier + ".json"
            )
            pathToConfigFile = (
                os.path.dirname(os.path.abspath(__file__)) + relConfigFilePath
            )
            eval = InitialLocalizationEvaluation(pathToConfigFile)

            for frame in framesToEvaluate:
                initializationResult = {}
                # load data
                pointCloud = eval.getPointCloud(frame, dataSetPath)
                Y = pointCloud[0][::10]

                # perform l1 skeletonization
                # visualize result
                # save result

                # perform som
                # visualize result
                # save result

                # extract minimum spanning tree
                # visualize result
                minSpanTree = minimalSpanningTree(distance_matrix(Y, Y))
                longestPaths = find_n_longest_paths(minSpanTree, 3)

                print(longestPaths)
                fig, ax = setupLatexPlot3D()
                l = [item for sublist in longestPaths for item in sublist]
                adjacencyMatrix = np.zeros((Y.shape[0], Y.shape[0]))
                for path in longestPaths:
                    predecessorIdxs = path[1:]
                    nodeIdxs = path[:-1]
                    for nodeIdx, predecessorIdx in zip(nodeIdxs, predecessorIdxs):
                        adjacencyMatrix[nodeIdx, predecessorIdx] = 1
                plotGraph(ax=ax, X=Y, adjacencyMatrix=adjacencyMatrix)
                plotGraph(
                    ax=ax,
                    X=Y,
                    adjacencyMatrix=(minSpanTree != 0).astype(int),
                    lineColor=[1, 0, 0],
                )
                plt.show()
                # adjacencyMatrix = (minSpanTree != 0).astype(int)
                # fig, ax = setupLatexPlot3D()
                # plotGraph(ax=ax, X=Y, adjacencyMatrix=adjacencyMatrix)

                # dist_matrix, predecessors = johnson(
                #     csgraph=minSpanTree, directed=False, return_predecessors=True
                # )

                # distant_idxs = np.unravel_index(
                #     np.argmax(dist_matrix, axis=None), dist_matrix.shape
                # )
                # plotPoint(ax=ax, x=Y[distant_idxs[0], :], color=[1, 0, 0])
                # plotPoint(ax=ax, x=Y[distant_idxs[1], :], color=[1, 0, 0])

                # # remove all points from the longest path from the graph
                # pathIndices = []
                # idx = distant_idxs[1]
                # while idx != distant_idxs[0]:
                #     idx = predecessors[distant_idxs[0], idx]
                #     pathIndices.append(idx)
                # plt.show(block=True)

                # subgraph = minSpanTree.copy()
                # for rowIdx in pathIndices:
                #     for colIdx in pathIndices:
                #         subgraph[rowIdx, colIdx] = 0
                # # subgraph[pathIndices, :] = 0  # set rows to zero
                # # subgraph[:, pathIndices] = 0  # set columns to zero

                # fig, ax = setupLatexPlot3D()
                # subGraphAdjacencyMatrix = (subgraph != 0).astype(int)
                # plotGraph(ax=ax, X=Y, adjacencyMatrix=subGraphAdjacencyMatrix)

                # dist_matrix, predecessors = johnson(
                #     csgraph=subgraph, directed=False, return_predecessors=True
                # )
                # dist_matrix[np.isinf(dist_matrix)] = 0
                # distant_idxs = np.unravel_index(
                #     np.argmax(dist_matrix, axis=None), dist_matrix.shape
                # )
                # plotPoint(ax=ax, x=Y[distant_idxs[0], :], color=[1, 0, 0])
                # plotPoint(ax=ax, x=Y[distant_idxs[1], :], color=[1, 0, 0])
                # plt.show(block=True)
                # save result

                # perform correspondance assignment
                # visualize result
                # save result

                # perform inverse kinematics
                # visualize result
                # save result

                # perform tracking
                # visualize result
                # save result
