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
    from src.evaluation.graspingAccuracy.graspingAccuracyEvaluation import (
        GraspingAccuracyEvaluation,
    )
    from src.tracking.registration import NonRigidRegistration
    from src.tracking.cpd.cpd import CoherentPointDrift
    from src.tracking.spr.spr import StructurePreservedRegistration
    from src.tracking.kpr.kpr import (
        KinematicsPreservingRegistration,
        KinematicsModelDart,
    )
    from src.visualization.dartVisualizer import DartVisualizer, DartScene
except:
    print("Imports for plotting kinematic regulatization for tracking chapter failed.")
    raise
runOpt = {
    "runInitialization": False,
    "saveInitializationResult": True,
    "saveRegistrationResults": True,
    "runKPR": False,
    "runStiffness": True,
    "runGravity": False,
    "runServoConstraints": True,
    "physicsSimulation": False,
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
# filePath_unoccluded = "data/darus_data_download/data/20230522_RoboticWireHarnessMounting/20230522_140014_arena/data/20230522_140238_164143_image_rgb.png"
filePath_unoccluded = "data/darus_data_download/data/20230522_RoboticWireHarnessMounting/20230522_141025_arena/data/20230522_141443_489686_image_rgb.png"

# filePath_occluded = "data/darus_data_download/data/20230516_Configurations_labeled/20230516_115857_arena/data/20230516_120419_158610_image_rgb.png"
# filePath_occluded = "data/darus_data_download/data/20230522_RoboticWireHarnessMounting/20230522_140014_arena/data/20230522_140313_145359_image_rgb.png"
filePath_occluded = "data/darus_data_download/data/20230522_RoboticWireHarnessMounting/20230522_141025_arena/data/20230522_141433_852919_image_rgb.png"

# filePath_RobotState = "data/darus_data_download/data/20230522_RoboticWireHarnessMounting/20230522_141025_arena/data/20230522_141433_852919_robot_state.json"
# filePath_RobotState = "data/darus_data_download/data/20230522_RoboticWireHarnessMounting/20230522_140014_arena/data/20230522_140313_145359_robot_state.json"
filePath_RobotState = "data/darus_data_download/data/20230522_RoboticWireHarnessMounting/20230522_141025_arena/data/20230522_141433_852919_robot_state.json"


if __name__ == "__main__":
    # load point set
    eval = TrackingEvaluation(evalConfigPath)
    fileName_unoccluded = os.path.basename(filePath_unoccluded)
    dataSetFolderPath = os.path.dirname(os.path.dirname(filePath_unoccluded)) + "/"
    (Y, Y_colors) = eval.getPointCloud(fileName_unoccluded, dataSetFolderPath)
    fileName_occluded = os.path.basename(filePath_occluded)
    (Y_occluded, _) = eval.getPointCloud(fileName_occluded, dataSetFolderPath)
    pointCloudProcessor = PointCloudProcessing()
    boundingBoxParameters = {
        "xMin": -0.35,
        "xMax": 0.45,
        "yMin": 0.15,
        "yMax": 0.25,
        "zMin": 0,
        "zMax": 2,
    }
    occlusion_mask = pointCloudProcessor.getMaskFromBoundingBox(
        Y,
        boundingBoxParameters["xMin"],
        boundingBoxParameters["xMax"],
        boundingBoxParameters["yMin"],
        boundingBoxParameters["yMax"],
        boundingBoxParameters["zMin"],
        boundingBoxParameters["zMax"],
    )
    Y_occluded = Y[np.invert(occlusion_mask), :]

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

    q0 = model.getGeneralizedCoordinates()
    q0[model.getBranchRootDofIndices(5)[0]] = -0.9
    q0[model.getBranchRootDofIndices(3)[0]] = -1.5

    # servo constraints
    robotState = eval.loadRobotState(filePath=filePath_RobotState)
    graspEval = GraspingAccuracyEvaluation()
    releasePositionFileIndices = [2, 5, 8]
    releasePositions = []
    for releasePositionFileIndex in releasePositionFileIndices:
        releasePositions.append(
            graspEval.loadGroundTruthGraspingPose(
                dataSetFolderPath=dataSetFolderPath,
                fileNumber=releasePositionFileIndex,
            )[1]
        )
    constaintPositions = releasePositions
    constaintPositions.append(
        np.array(robotState["O_T_EE"][-4:-1]) + np.array([0, 0, 0.0])
    )

    constaintNodeIndices = []
    for i in range(0, 4):
        graspingLocalCoordinate = graspEval.loadGraspingLocalCoordinates(
            dataSetFolderPath
        )[i]
        constrainedNodeIndex = model.getBodyNodeIndexFromBranchLocalCoodinate(
            branchIndex=graspingLocalCoordinate[0], s=graspingLocalCoordinate[1]
        )
        constaintNodeIndices.append(constrainedNodeIndex)

    if runOpt["runKPR"]:
        kprParameters = {
            "max_iterations": 100,
            "wStiffness": 0,
            "stiffnessAnnealing": 0.97,
            "ik_iterations": 10,
            "damping": 1,
            "dampingAnnealing": 0.9,
            "minDampingFactor": 0.1,
            "mu": 0.01,
            "sigma2": 0.01,
            "normalize": 0,
            "q0": q0,
        }
        kpr = KinematicsPreservingRegistration(
            Y=Y_occluded,
            qInit=initializationResult["localization"]["qInit"],
            model=kinematicModel,
            **kprParameters,
        )
        kpr.registerCallback(eval.getVisualizationCallback(kpr))
        # kpr.register()
        registrationResult_kpr = eval.runRegistration(
            registration=kpr, checkConvergence=False
        )

    if runOpt["runStiffness"]:
        # stiffness regularizaiton
        kprParameters_stiffness = {
            "max_iterations": 100,
            "wStiffness": 100,
            "stiffnessAnnealing": 0.93,
            "ik_iterations": 10,
            "damping": 1,
            "dampingAnnealing": 0.97,
            "minDampingFactor": 0.3,
            "mu": 0.01,
            "normalize": 0,
            "sigma2": 0.01,
            "gravity": np.array([0, 0, 9.81]),
            "wGravity": 0,
            "gravitationalAnnealing": 1,
            "groundLevel": np.array([0, 0, 0.35]),
            "q0": q0,
            "wConstraint": 1000,
            "constrainedNodeIndices": constaintNodeIndices,
            "constrainedPositions": constaintPositions,
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
            "damping": 1,
            "dampingAnnealing": 1,
            "minDampingFactor": 0.1,
            "mu": 0.01,
            "normalize": 0,
            "sigma2": 0.01,
            "gravity": np.array([0, 0, 9.81]),
            "wGravity": 10,
            "gravitationalAnnealing": 1,
            "groundLevel": np.array([0, 0, -1]),
            "q0": model.getGeneralizedCoordinates(),
            "wConstraint": 1000,
            "constrainedNodeIndices": constaintNodeIndices,
            "constrainedPositions": constaintPositions,
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
    constaintPositions[-1] = constaintPositions[-1] + np.array([0, 0, 0.1])
    if runOpt["runServoConstraints"]:
        kprParameters_servo = {
            "max_iterations": 100,
            "wStiffness": 0,
            "stiffnessAnnealing": 0.97,
            "ik_iterations": 10,
            "damping": 1,
            "dampingAnnealing": 0.9,
            "minDampingFactor": 0.1,
            "mu": 0.01,
            "normalize": 0,
            "sigma2": 0.01,
            "gravity": np.array([0, 0, 9.81]),
            "wGravity": 0,
            "gravitationalAnnealing": 1,
            "groundLevel": np.array([0, 0, 0.35]),
            "wConstraint": 100,
            "constrainedNodeIndices": constaintNodeIndices,
            "constrainedPositions": constaintPositions,
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

    if runOpt["physicsSimulation"]:
        # servo constraints
        kprParameters_physics = {
            "max_iterations": 1,
            "wStiffness": 1,
            "stiffnessAnnealing": 0.9,
            "ik_iterations": 5,
            "damping": 1,
            "dampingAnnealing": 1,
            "minDampingFactor": 0.01,
            "mu": 0.01,
            "normalize": 0,
            "sigma2": 0.01,
        }
        kpr_physics = KinematicsPreservingRegistration(
            Y=Y,
            qInit=initializationResult["localization"]["qInit"],
            model=kinematicModel,
            **kprParameters_physics,
        )
        kpr_physics.registerCallback(eval.getVisualizationCallback(kpr_physics))
        # kpr.register()
        registrationResult_physics = eval.runRegistration(
            registration=kpr_physics, checkConvergence=False
        )
        # initialize dart simulation and visualize contact forces
        robotState = eval.loadRobotState(filePath=filePath_RobotState)
        q_robot = robotState["q"]
        q_robot.append(0)
        q_robot.append(0)
        model.setStiffnessForAllDof(0)
        dartScene = DartScene(
            skel=model.skel.clone(), q=registrationResult_physics["q"], skelAlpha=0.3
        )
        dartScene.robotSkel.setMobile(False)
        dartScene.setRobotPosition(robotState["q"])
        for step in range(0, 10):
            dartScene.world.step()
            contacts = dartScene.world.getLastCollisionResult().getContacts()
            if contacts is not None:
                contactPoints = []
                contactNormals = []
                for i in range(0, len(contacts)):
                    contactPoints.append(contacts[i].point)
                    contactNormals.append(contacts[i].normal)
                dartScene.addPointCloud(contactPoints)
            dartScene.showFrame()
    print("Done.")
