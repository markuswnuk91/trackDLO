import sys
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import numpy as np
import matplotlib.lines as mlines
from scipy.stats import multivariate_normal

try:
    sys.path.append(os.getcwd().replace("/plot", ""))
    from src.sensing.preProcessing import PointCloudProcessing
    from src.visualization.plot3D import *
    from src.visualization.plot2D import *
    from src.evaluation.tracking.trackingEvaluation import TrackingEvaluation
    from src.tracking.registration import NonRigidRegistration
    from src.tracking.cpd.cpd import CoherentPointDrift
    from src.tracking.spr.spr import StructurePreservedRegistration
    from src.tracking.kpr.kpr import (
        KinematicsPreservingRegistration,
        KinematicsModelDart,
    )
except:
    print("Imports for plotting kinematic regulatization for tracking chapter failed.")
    raise
runOpt = {
    "runInitialization": True,
    "runRegistrations": False,
    "saveInitializationResult": True,
    "saveRegistrationResults": True,
    "runStiffness": True,
    "runGravity": True,
    "runServoConstraints": True,
}
visOpt = {"visualizeInitialLocalizationResult": False}
saveOpt = {
    "savePlots": False,
    "initializationResultPath": "data/plots/physicalPlausibility",
    "saveFolderPath": "imgs/physicalPlausibility",
    "dpi": 300,
}
styleOpt = {"branchColorMap": thesisColorPalettes["viridis"]}
n_th_element = 10
nodeSize = 30
edgeSize = 2
evalConfigPath = "plot/plotTracking/config.json"
# filePath_unoccluded = "data/darus_data_download/data/20230516_Configurations_labeled/20230516_115857_arena/data/20230516_120411_389179_image_rgb.png"
filePath_unoccluded = "data/darus_data_download/data/20230522_RoboticWireHarnessMounting/20230522_140014_arena/data/20230522_140238_164143_image_rgb.png"
# filePath_occluded = "data/darus_data_download/data/20230516_Configurations_labeled/20230516_115857_arena/data/20230516_120419_158610_image_rgb.png"
filePath_occluded = "data/darus_data_download/data/20230522_RoboticWireHarnessMounting/20230522_140014_arena/data/20230522_140313_145359_image_rgb.png"


def determineCorrespondanceColors(model, Y, P=None):
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
    X_colors = []
    for n in bodyNodeIndices:
        branchIndex = correspondingBranches[n]
        nodeIndicesInBranchList = model.getBranchBodyNodeIndices(branchIndex)
        nodeIndexInBranch = np.where(np.array(nodeIndicesInBranchList) == n)[0]
        nodeColor = branchColorMaps[branchIndex].to_rgba(
            nodeIndexInBranch[0] / len(nodeIndicesInBranchList)
        )
        X_colors.append(nodeColor)
    if P is not None:
        Y_colors = []
        for m, y in enumerate(Y):
            y_color = np.zeros(4)
            for n in bodyNodeIndices:
                y_color += P[n, m] * np.array(list(X_colors[n]))
            y_color += (1 - np.sum(P[:, m])) * np.array([0.5, 0.5, 0.5, 0.9])
            Y_colors.append(y_color)
        return X_colors, Y_colors
    else:
        return X_colors


def colorBranchesWithLinearColorMap(model, Y, P):
    bodyNodeIndices, correspondingBranches = (
        model.getBranchCorrespondancesForBodyNodes()
    )
    branchColorMap = styleOpt["branchColorMap"]
    bodyNodeIndices, correspondingBranches = (
        model.getBranchCorrespondancesForBodyNodes()
    )
    # branchColors = []
    # for branchIndex in range(0, model.getNumBranches()):
    #     branchColor = styleOpt["branchColorMap"].to_rgba(
    #         branchIndex / (model.getNumBranches() - 1)
    #     )
    #     branchColors.append(branchColor[:3])
    # branchColorMaps = []
    # for i, branchColor in enumerate(branchColors):
    #     branchColor = np.array(branchColor + (1,))
    #     endColor = branchColor + 0.7 * (np.array([1, 1, 1, 1]) - branchColor)
    #     branchColorMap = LinearSegmentedColormap.from_list(
    #         "branchColorMap_i", [branchColor, endColor]
    #     )
    #     norm = Normalize(vmin=0, vmax=1)  # Normalize values between 0 and 1
    #     mapper = ScalarMappable(norm=norm, cmap=branchColorMap)
    #     branchColorMaps.append(mapper)
    X_colors = []
    for n in bodyNodeIndices:
        branchIndex = correspondingBranches[n]
        nodeIndicesInBranchList = model.getBranchBodyNodeIndices(branchIndex)
        nodeIndexInBranch = np.where(np.array(nodeIndicesInBranchList) == n)[0]
        if model.isOuterBranch(model.getBranch(branchIndex)):
            if branchIndex != 0:
                nodeColor = branchColorMap.to_rgba(
                    nodeIndexInBranch[0] / (len(nodeIndicesInBranchList) - 1)
                )
            else:
                nodeColor = branchColorMap.to_rgba(
                    (len(nodeIndicesInBranchList) - nodeIndexInBranch[0])
                    / len(nodeIndicesInBranchList)
                )
        else:
            nodeColor = branchColorMap.to_rgba(0)
        X_colors.append(nodeColor)
    if P is not None:
        Y_colors = []
        for m, y in enumerate(Y):
            y_color = np.zeros(4)
            for n in bodyNodeIndices:
                y_color += P[n, m] * np.array(list(X_colors[n]))
            y_color += (1 - np.sum(P[:, m])) * np.array([0.5, 0.5, 0.5, 0.9])
            Y_colors.append(y_color)
        return np.array(X_colors)[:, :3], np.array(Y_colors)[:, :3]
    else:
        return np.array(X_colors)[:, :3]


if __name__ == "__main__":
    # load point set
    eval = TrackingEvaluation(evalConfigPath)
    fileName_unoccluded = os.path.basename(filePath_unoccluded)
    dataSetFolderPath = os.path.dirname(os.path.dirname(filePath_unoccluded)) + "/"
    (Y, Y_colors) = eval.getPointCloud(fileName_unoccluded, dataSetFolderPath)
    fileName_occluded = os.path.basename(filePath_occluded)
    (Y_occluded, _) = eval.getPointCloud(fileName_occluded, dataSetFolderPath)
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

    pointCloudProcessor = PointCloudProcessing()
    boundingBoxParameters = {
        "xMin": -0.5,
        "xMax": 1,
        "yMin": 0,
        "yMax": 1,
        "zMin": 0,
        "zMax": 2,
    }
    Y_occluded = pointCloudProcessor.filterPointsBoundingBox(
        Y,
        boundingBoxParameters["xMin"],
        boundingBoxParameters["xMax"],
        boundingBoxParameters["yMin"],
        boundingBoxParameters["yMax"],
        boundingBoxParameters["zMin"],
        boundingBoxParameters["zMax"],
    )
    modelParameters = initializationResult["modelParameters"]
    model = eval.generateModel(modelParameters)
    X_init = model.getCartesianBodyCenterPositions()
    if visOpt["visualizeInitialLocalizationResult"]:
        fig, ax = setupLatexPlot3D()
        plotPointSet(ax=ax, X=Y_occluded, color=[1, 0, 0], alpha=0.1, size=1)
        plt.show(block=False)
        plotPointSet(
            ax=ax,
            X=X_init,
            size=10,
            markerStyle="o",
            edgeColor=[0, 0, 1],
            color=[1, 1, 1],
        )
        scale_axes_to_fit(ax=ax, points=X_init)
        ax.view_init(elev=90, azim=0)
        plt.show(block=True)
    kinematicModel = KinematicsModelDart(model.skel.clone())

    if runOpt["runStiffness"]:
        # stiffness regularizaiton
        kprParameters_stiffness = {
            "max_iterations": 100,
            "wStiffness": 100,
            "stiffnessAnnealing": 1,
            "ik_iterations": 10,
            "damping": 0.01,
            "dampingAnnealing": 0.7,
            "minDampingFactor": 0.01,
            "mu": 0.01,
            "normalize": 0,
            "sigma2": 0.001,
        }
        kpr_stiffness = KinematicsPreservingRegistration(
            Y=Y_occluded,
            qInit=initializationResult["localization"]["qInit"],
            model=kinematicModel,
            **kprParameters_stiffness,
        )
        kpr_stiffness.registerCallback(eval.getVisualizationCallback(kpr_stiffness))
        # kpr.register()
        registrationResult_stiffness = eval.runRegistration(
            registration=kpr_stiffness, checkConvergence=False
        )

    if runOpt["runGravity"]:
        # gravitational regularization
        kprParameters_gravity = {
            "max_iterations": 100,
            "wStiffness": 10,
            "stiffnessAnnealing": 0.9,
            "ik_iterations": 10,
            "damping": 0.1,
            "dampingAnnealing": 0.7,
            "minDampingFactor": 0.01,
            "mu": 0.01,
            "normalize": 0,
            "sigma2": 0.01,
            "gravity": np.array([0, 0, -9.81]),
            "wGravity": 1,
            "gravitationalAnnealing": 1,
        }
        kpr_gravity = KinematicsPreservingRegistration(
            Y=Y_occluded,
            qInit=initializationResult["localization"]["qInit"],
            model=kinematicModel,
            **kprParameters_gravity,
        )
        kpr_gravity.registerCallback(eval.getVisualizationCallback(kpr_gravity))
        # kpr.register()
        registrationResult_gravitys = eval.runRegistration(
            registration=kpr_gravity, checkConvergence=False
        )

    if runOpt["runServoConstraints"]:
        # servo constraints
        kprParameters_servo = {
            "max_iterations": 100,
            "wStiffness": 1,
            "stiffnessAnnealing": 1,
            "ik_iterations": 5,
            "damping": 1,
            "dampingAnnealing": 1,
            "minDampingFactor": 0.01,
            "mu": 0.01,
            "normalize": 0,
            "sigma2": 0.01,
            "gravitationalAnnealing": 1,
            "wConstraint": 100,
            "constrainedNodeIndices": [33],
            "constrainedPositions": [np.array([0, 0, 1])],
        }
        kpr_servo = KinematicsPreservingRegistration(
            Y=Y_occluded,
            qInit=initializationResult["localization"]["qInit"],
            model=kinematicModel,
            **kprParameters_servo,
        )
        kpr_servo.registerCallback(eval.getVisualizationCallback(kpr_servo))
        # kpr.register()
        registrationResult_servo = eval.runRegistration(
            registration=kpr_servo, checkConvergence=False
        )
