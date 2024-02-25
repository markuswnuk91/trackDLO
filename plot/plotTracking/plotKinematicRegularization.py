import sys
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import numpy as np

try:
    sys.path.append(os.getcwd().replace("/plot", ""))
    from src.visualization.plot3D import *
    from src.visualization.plot2D import *
    from src.evaluation.tracking.trackingEvaluation import TrackingEvaluation
    from src.tracking.cpd.cpd import CoherentPointDrift
    from src.tracking.kpr.kpr import (
        KinematicsPreservingRegistration,
        KinematicsModelDart,
    )
except:
    print("Imports for plotting kinematic regulatization for tracking chapter failed.")
    raise
runOpt = {"saveInitializationResult": True, "runInitialization": False}
saveOpt = {
    "initializationResultPath": "data/plots/kinematicRegularization",
    "saveFolderPath": "imgs/kinematicRegularization",
    "dpi": 300,
}
styleOpt = {"branchColorMap": thesisColorPalettes["viridis"]}

evalConfigPath = "plot/plotTracking/config.json"
# filePath = "data/darus_data_download/data/20230516_Configurations_labeled/20230516_115857_arena/data/20230516_120351_790481_image_rgb.png"
filePath_unoccluded = "data/darus_data_download/data/20230516_Configurations_labeled/20230516_115857_arena/data/20230516_120411_389179_image_rgb.png"
# "data/darus_data_download/data/20230522_RoboticWireHarnessMounting/20230522_140014_arena/data/20230522_140157_033099_image_rgb.png"
filePath_occluded = "data/darus_data_download/data/20230516_Configurations_labeled/20230516_115857_arena/data/20230516_120419_158610_image_rgb.png"
# "data/darus_data_download/data/20230522_RoboticWireHarnessMounting/20230522_140014_arena/data/20230522_140355_166626_image_rgb.png"
if __name__ == "__main__":
    # load point set
    eval = TrackingEvaluation(evalConfigPath)
    fileName_unoccluded = os.path.basename(filePath_unoccluded)
    dataSetFolderPath = os.path.dirname(os.path.dirname(filePath_unoccluded)) + "/"
    (Y, _) = eval.getPointCloud(fileName_unoccluded, dataSetFolderPath)
    fig, ax = setupLatexPlot3D()
    plotPointSet(ax=ax, X=Y, color=[1, 0, 0], alpha=0.1, size=1)
    # mask = np.ones(len(Y), dtype=bool)
    # mask[300:1000] = False
    plt.show(block=False)
    # load occluded point set
    fileName_occluded = os.path.basename(filePath_occluded)
    (Y_occluded, _) = eval.getPointCloud(fileName_occluded, dataSetFolderPath)
    mask = np.ones(len(Y_occluded), dtype=bool)
    mask[50:300] = False
    Y_occluded = Y_occluded[mask]

    # get inital configuration
    if runOpt["runInitialization"]:
        initializationResult = eval.runInitialization(
            dataSetFolderPath, fileName_unoccluded
        )
        if runOpt["saveInitializationResult"]:
            eval.saveWithPickle(
                data=initializationResult,
                filePath=os.path.join(
                    saveOpt["initializationResultPath"],
                    "initializationResult.pkl",
                ),
                recursionLimit=10000,
            )
    else:
        # save correspondance estimation results
        initializationResult = eval.loadResults(
            os.path.join(
                saveOpt["initializationResultPath"],
                "initializationResult.pkl",
            )
        )
    X_init = initializationResult["localization"]["X"]
    modelParameters = initializationResult["modelParameters"]
    model = eval.generateModel(modelParameters)

    # visualize correspondances
    # rotate model
    model.setInitialPose(initialRotation=[np.pi / 2, 0, np.pi / 2])
    # translate model
    X = model.getCartesianBodyCenterPositions()
    offset = np.mean(Y, axis=0) - np.mean(X, axis=0)
    initialPosition = X[0, :] + offset
    model.setInitialPose(initialPosition=initialPosition)

    X_init = model.getCartesianBodyCenterPositions()
    plotPointSet(
        ax=ax, X=X_init, size=10, markerStyle="o", edgeColor=[0, 0, 1], color=[1, 1, 1]
    )
    scale_axes_to_fit(ax=ax, points=X_init)
    eval.config["cpdParameters"]["alpha"] = 0
    correspondanceRegistration = CoherentPointDrift(
        **{"X": X_init, "Y": Y}, **eval.config["cpdParameters"]
    )
    P = correspondanceRegistration.estimateCorrespondance()
    bodyNodeIndices, correspondingBranches = (
        model.getBranchCorrespondancesForBodyNodes()
    )
    branchColors = []
    for branchIndex in range(0, model.getNumBranches()):
        branchColor = styleOpt["branchColorMap"].to_rgba(
            branchIndex / (model.getNumBranches() - 1)
        )
        branchColors.append(branchColor[:3])

    branchColorMaps = []
    for i, branchColor in enumerate(branchColors):
        branchColor = np.array(branchColor + (1,))
        endColor = branchColor + 0.7 * (np.array([1, 1, 1, 1]) - branchColor)
        branchColorMap = LinearSegmentedColormap.from_list(
            "branchColorMap_i", [branchColor, endColor]
        )
        norm = Normalize(vmin=0, vmax=1)  # Normalize values between 0 and 1
        mapper = ScalarMappable(norm=norm, cmap=branchColorMap)
        branchColorMaps.append(mapper)

    Y_colors = []
    X_colors = []
    for n in bodyNodeIndices:
        branchIndex = correspondingBranches[n]
        nodeIndicesInBranchList = model.getBranchBodyNodeIndices(branchIndex)
        nodeIndexInBranch = np.where(np.array(nodeIndicesInBranchList) == n)[0]
        nodeColor = branchColorMaps[branchIndex].to_rgba(
            nodeIndexInBranch[0] / len(nodeIndicesInBranchList)
        )
        X_colors.append(nodeColor)

    for m, y in enumerate(Y):
        y_color = np.zeros(4)
        for n in bodyNodeIndices:
            y_color += P[n, m] * np.array(list(X_colors[n]))
        y_color += (1 - np.sum(P[:, m])) * np.array([0.5, 0.5, 0.5, 0.9])
        Y_colors.append(y_color)
    plotPointSet(ax=ax, X=Y, color=Y_colors, size=1)
    plotPointSet(ax=ax, X=X_init, color=[1, 1, 1], edgeColor=X_colors)

    # run tracking (GMM)
    eval.config["cpdParameters"]["alpha"] = 1
    cpd = CoherentPointDrift(
        **{"X": X_init, "Y": Y_occluded}, **eval.config["cpdParameters"]
    )
    P = cpd.estimateCorrespondance()
    cpd.registerCallback(eval.getVisualizationCallback(cpd))
    cpd.register()

    # run tracking Regularized
    q_init = initializationResult["localization"]["q"]
    model.setGeneralizedCoordinates(q_init)
    kinematicModel = KinematicsModelDart(model.skel.clone())
    kpr = KinematicsPreservingRegistration(
        Y=Y_occluded, qInit=q_init, model=kinematicModel, **eval.config["kprParameters"]
    )
    kpr.registerCallback(eval.getVisualizationCallback(kpr))
    kpr.register()

    fig, ax = setupLatexPlot3D()
    plotPointSet(ax=ax, X=kpr.T, size=30, color=[1, 1, 1], edgeColor=[0, 0, 1])
    plotPointSet(ax=ax, X=Y_occluded, size=1, color=[1, 0, 0])
    plotGraph3D(ax=ax, X=kpr.T, adjacencyMatrix=model.getBodyNodeNodeAdjacencyMatrix())
    scale_axes_to_fit(points=kpr.T)

    fig, ax = setupLatexPlot3D()
    plotPointSet(ax=ax, X=cpd.T, size=30, color=[1, 1, 1], edgeColor=[0, 0, 1])
    plotGraph3D(ax=ax, X=cpd.T, adjacencyMatrix=model.getBodyNodeNodeAdjacencyMatrix())
    plotPointSet(ax=ax, X=Y_occluded, size=1, color=[1, 0, 0])
    scale_axes_to_fit(ax=ax, points=cpd.T)

    # visualizat point set
    plt.show(block=True)
    print("Done.")
