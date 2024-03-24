import sys
import os
import numpy as np
import dartpy as dart
from scipy.spatial.transform import Rotation as R
from PIL import Image

try:
    sys.path.append(os.getcwd().replace("/plot", ""))
    from src.visualization.dartVisualizer import DartVisualizer, DartScene
    from src.evaluation.graspingAccuracy.graspingAccuracyEvaluation import (
        GraspingAccuracyEvaluation,
    )
    from src.tracking.kpr.kpr import (
        KinematicsPreservingRegistration,
        KinematicsModelDart,
    )
    from src.simulation.forceUpdate import ForceUpdate
    from src.sensing.dataHandler import DataHandler
except:
    print("Imports for plotting Dynamics Simulation failed.")
    raise

runOpt = {
    "runInitialization": False,
    "saveInitializationResult": True,
    "runRegistration": False,
    "saveRegistrationResult": True,
    "runPointCloudVisualization": False,
    "runContactVisualization": True,
}

saveOpt = {
    "savePlots": False,
    "initializationResultPath": "data/plots/physicsSimulation",
    "saveFolderPath": "imgs/physicsSimulation",
    "dpi": 300,
}

filePath_unoccluded = "data/darus_data_download/data/20230522_RoboticWireHarnessMounting/20230522_141025_arena/data/20230522_141108_447486_image_rgb.png"
filePath_occluded = "data/darus_data_download/data/20230522_RoboticWireHarnessMounting/20230522_141025_arena/data/20230522_141433_852919_image_rgb.png"
filePath_RobotState = "data/darus_data_download/data/20230522_RoboticWireHarnessMounting/20230522_141025_arena/data/20230522_141433_852919_robot_state.json"
evalConfigPath = "plot/plotTracking/config.json"
setupDescriptionPath = "data/darus_data_download/data/20230522_RoboticWireHarnessMounting/20230522_141025_arena/setup.json"

camEye = [0.35, 1.7, 1.7]
camCenter = [0.35, 0, 0]
camUp = [0, 0, 1]
refImg = Image.open(saveOpt["saveFolderPath"] + "/realScene.png")
width = refImg.size[0]
height = refImg.size[1]


def setupDartScene(skelAlpha=None, robotAlpha=None):
    skelAlpha = 0.3 if skelAlpha is None else skelAlpha
    robotAlpha = 0.5 if robotAlpha is None else robotAlpha
    robotState = eval.loadRobotState(filePath=filePath_RobotState)
    q_robot = robotState["q"]
    q_robot.append(0)
    q_robot.append(0)
    model.setStiffnessForAllDof(0)
    model.setDampingForAllDof(0)
    model.setColor([0.3, 0.3, 1])
    q = registrationResult["q"]
    dartScene = DartScene(
        skel=model.skel.clone(),
        q=q,
        skelAlpha=skelAlpha,
        robotAlpha=robotAlpha,
        loadCell=False,
    )
    dataHandler = DataHandler()
    setupDesciption = dataHandler.loadFromJson(setupDescriptionPath)
    fixturePositions = []
    fixtureOffset = np.array([0, 0, -0.1])
    fixtureRotations_Z = [0, np.pi / 2, 0, 0, np.pi / 2, np.pi / 2]
    for desciption in setupDesciption["fixtures"]:
        robotPose = np.reshape(
            np.array(desciption["robotPoseAtFixturePosition"]), (4, -1), order="F"
        )
        fixturePositions.append(robotPose[:3, 3] + fixtureOffset)

    for fixturePosition, fixtureRotation_Z in zip(fixturePositions, fixtureRotations_Z):
        dartScene.loadFixture(
            x=fixturePosition[0],
            y=fixturePosition[1],
            z=fixturePosition[2],
            rz=fixtureRotation_Z,
            alpha=0.7,
        )

    dartScene.robotSkel.setMobile(False)
    dartScene.setRobotPosition(robotState["q"])

    # dartScene.setCameraPosition(eye=[3, 0.3, 2], center=[0, 0, 0], up=[0, 0, 1])
    dartScene.setCameraPosition(eye=camEye, center=camCenter, up=camUp)
    return dartScene


if __name__ == "__main__":
    # load point set
    eval = GraspingAccuracyEvaluation(evalConfigPath)
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

    modelParameters = initializationResult["modelParameters"]
    model = eval.generateModel(modelParameters)

    if runOpt["runRegistration"]:
        X_init = model.getCartesianBodyCenterPositions()
        kinematicModel = KinematicsModelDart(model.skel.clone())
        kprParameters = {
            "max_iterations": 100,
            "wStiffness": 10000,
            "stiffnessAnnealing": 0.97,
            "ik_iterations": 10,
            "damping": 1,
            "dampingAnnealing": 1,
            "minDampingFactor": 0.1,
            "mu": 0.01,
            "normalize": 0,
            "sigma2": 0.01,
            "gravity": np.array([0, 0, 9.81]),
            "wGravity": 0.1,
            "gravitationalAnnealing": 1,
            "groundLevel": np.array([0, 0, 0.35]),
            # "q0": q0,
        }
        kpr = KinematicsPreservingRegistration(
            Y=Y_occluded,
            qInit=initializationResult["localization"]["qInit"],
            model=kinematicModel,
            **kprParameters,
        )
        kpr.registerCallback(eval.getVisualizationCallback(kpr))
        # kpr.register()
        registrationResult = eval.runRegistration(
            registration=kpr, checkConvergence=False
        )
        if runOpt["saveRegistrationResult"]:
            eval.saveWithPickle(
                data=registrationResult,
                filePath=os.path.join(
                    saveOpt["initializationResultPath"],
                    "registrationResult.pkl",
                ),
                recursionLimit=10000,
            )
    else:
        # save registration result
        registrationResult = eval.loadResults(
            os.path.join(
                saveOpt["initializationResultPath"],
                "registrationResult.pkl",
            )
        )
    # initialize dart simulation and visualize contact forces
    releasePositionFileIndices = [2, 5, 8, 11]
    releasePositions = []
    for releasePositionFileIndex in releasePositionFileIndices:
        releasePositions.append(
            eval.loadGroundTruthGraspingPose(
                dataSetFolderPath=dataSetFolderPath,
                fileNumber=releasePositionFileIndex,
            )[1]
        )
    constaintPositions = releasePositions
    constaintNodeIndices = []
    for i in range(0, 4):
        graspingLocalCoordinate = eval.loadGraspingLocalCoordinates(dataSetFolderPath)[
            i
        ]
        constrainedNodeIndex = model.getBodyNodeIndexFromBranchLocalCoodinate(
            branchIndex=graspingLocalCoordinate[0], s=graspingLocalCoordinate[1]
        )
        constaintNodeIndices.append(constrainedNodeIndex)
    # add another constraint
    constaintNodeIndices.append(constaintNodeIndices[-1] - 1)
    constaintPositions.append(constaintPositions[-1] + np.array([0, -0.05, 0]))

    if runOpt["runPointCloudVisualization"]:
        pointCloudApha = 1
        pointCloudColor = [1, 0, 0]
        dartScene = setupDartScene(skelAlpha=0.3, robotAlpha=0.5)
        forceUpdate = ForceUpdate(
            dartSkel=model.skel.clone(), Kp=0.1, Kd=0.1, forceLimit=0.1
        )
        dartScene.addPointCloud(points=Y_occluded, colors=[1, 0, 0], size=0.003)
        for target in registrationResult["T"]:
            dartScene.addSphere(radius=0.008, color=[0, 0, 1], offset=target)
        # dartScene.addPointCloud(
        #     points=registrationResult["T"], colors=[0, 0, 1], size=0.01
        # )

        if saveOpt["savePlots"]:
            dartScene.showFrame()
            saveFilePath = (
                saveOpt["saveFolderPath"] + "/impedanceControllerVisualization"
            )

            dartScene.saveFrame(
                saveFilePath,
                eye=camEye,
                center=camCenter,
                up=camUp,
                width=width,
                height=height,
            )
        for step in range(0, 1000):
            dartScene.showFrame()

    if runOpt["runContactVisualization"]:
        pointCloudColor = [1, 0.1, 0.1]
        pointCloudApha = 1
        dartScene = setupDartScene()
        forceUpdate = ForceUpdate(
            dartSkel=model.skel.clone(), Kp=10, Kd=1, forceLimit=100000
        )

        contactPointCloudShape = dart.dynamics.PointCloudShape(0.008)
        # Since contact points may change during execution, dynamic data variance
        # is assumed for the pointcloud of contacts. Otherwise, OSG will not render
        # the new points.
        contactPointCloudShape.setDataVariance(dart.dynamics.Shape.DataVariance.DYNAMIC)
        pointCloudSimpleFrame = dart.dynamics.SimpleFrame(
            dart.dynamics.Frame.World(), "ContactsVisualization"
        )
        pointCloudSimpleFrame.setShape(contactPointCloudShape)
        pcVisualAspect = pointCloudSimpleFrame.createVisualAspect()
        pcVisualAspect.setRGBA(
            [pointCloudColor[0], pointCloudColor[1], pointCloudColor[2], pointCloudApha]
        )
        dartScene.world.addSimpleFrame(pointCloudSimpleFrame)

        q_old = dartScene.skel.getPositions().copy()
        # dartScene.skel.setPositions(np.zeros(dartScene.skel.getNumDofs()))
        qd = registrationResult["q"]
        qd[5] -= 0.3
        for step in range(0, 1000):
            q = dartScene.skel.getPositions().copy()
            q_dot = dartScene.skel.getVelocities().copy()
            q_ddot = dartScene.skel.getAccelerations().copy()
            qd_dot = np.zeros(dartScene.skel.getNumDofs())
            qd_ddot = np.zeros(dartScene.skel.getNumDofs())
            tauExt = forceUpdate.computeExternalForceUpdateInGeneralizedCoordinates(
                q,
                q_dot,
                q_ddot,
                qd,
                qd_dot,
                qd_ddot,
                q_old=q_old,
                skel=dartScene.skel,
                method="StablePD",
            )
            q_old = q.copy()
            dartScene.skel.setForces(tauExt)
            # tauExt[:6] = np.zeros(6)
            # tauExt -= dartScene.skel.getGravityForces()
            # for i, dof in enumerate(dartScene.skel.getDof()):
            #     dof.setForce(float(0.0))
            # for i in range(6, dartScene.skel.getNumDofs() - 6):
            #     dof = dartScene.skel.getDof(i)
            #     dof.setForce(tauExt[i])
            # dartScene.skel.setForces(dartScene.skel.getCoriolisAndGravityForces()
            # dartScene.skel.setForces(tauExt)

            # tauImp = np.zeros(dartScene.skel.getNumDofs())
            # tauImp += dartScene.skel.getCoriolisAndGravityForces()
            # tauImp -= dartScene.skel.getGravityForces()
            # dartScene.skel.setForces(tauImp)
            # # do not compensate gravity to visualize contacts
            # # q = dartScene.skel.getPositions()
            # # q += dartScene.skel.getVelocities() * dartScene.skel.getTimeStep()
            # # qError = qd - q
            # # Kp = 1000
            # # tauImp += dartScene.skel.getMassMatrix() @ (Kp * qError)
            # # tauImp += dartScene.skel.getMassMatrix() @ dartScene.skel.getAccelerations()
            # for bodyNode, target in zip(
            #     dartScene.skel.getBodyNodes(),
            #     registrationResult["T"] + np.array([0, 0, -0.1]),
            # ):
            #     Kp = 100
            #     Kd = 10
            #     pos = bodyNode.getTransform().translation()
            #     vel = bodyNode.getLinearVelocity()
            #     pos += vel * dartScene.skel.getTimeStep()
            #     acc = bodyNode.getLinearAcceleration()
            #     vel += dartScene.skel.getLinearJacobian(bodyNode) @ (
            #         dartScene.skel.getInvMassMatrix()
            #         @ (
            #             -dartScene.skel.getCoriolisAndGravityForces()
            #             + dartScene.skel.getExternalForces()
            #         )
            #         * dartScene.skel.getTimeStep()
            #     )
            #     f = (
            #         dartScene.skel.getLinearJacobian(bodyNode)
            #         @ dartScene.skel.getMassMatrix()
            #         @ dartScene.skel.getLinearJacobian(bodyNode).T
            #         @ (Kp * (target - pos) - Kd * vel)
            #     )
            #     # f = Kp * (target - pos) - Kd * vel
            #     tauImp += dartScene.skel.getMassMatrix() @ (
            #         dartScene.skel.getLinearJacobian(bodyNode).T @ f
            #     )
            #     # tauImp += dartScene.skel.getLinearJacobian(bodyNode).T @ f

            #     bodyNode.addExtForce(f)

            # q = (
            #     dartScene.skel.getPositions()
            #     + dartScene.skel.getVelocities() * dartScene.skel.getTimeStep()
            # )
            # q_ddot = np.linalg.inv(
            #     (
            #         dartScene.skel.getMassMatrix()
            #         + Kd * dartScene.skel.getTimeStep() * np.eye(len(q))
            #     )
            # ) @ (
            #     -dartScene.skel.getCoriolisForces()
            #     - Kp * (q - qd)
            #     - Kd * dartScene.skel.getVelocities()
            #     - dartScene.skel.getExternalForces()
            # )
            # tauImp = Kp * (qd - q) - (
            #     Kd * dartScene.skel.getVelocities()
            #     + q_ddot * dartScene.skel.getTimeStep()
            # )
            # dartScene.skel.setForces(tauImp)

            for i, bodyNodeIndex in enumerate(constaintNodeIndices):
                bn = dartScene.skel.getBodyNode(bodyNodeIndex)
                desired_postion = constaintPositions[i]
                current_position = bn.getTransform().translation()
                current_veclocity = bn.getTransform().translation()
                pd_force = (
                    30 * (desired_postion - current_position)
                    - 1 * bn.getCOMLinearVelocity()
                )
                if np.linalg.norm(pd_force) > 100:
                    pd_force = np.sign(pd_force) * 100
                bn.addExtForce(pd_force)
            dartScene.world.step()
            contacts = dartScene.world.getLastCollisionResult().getContacts()
            contactPoints = []
            contactNormals = []
            if len(dartScene.arrowList) > 0:
                for arrow in dartScene.arrowList:
                    dartScene.removeArrow(arrow)
            for j in range(0, len(contacts)):
                contactPoints.append(contacts[j].point)
                contactNormals.append(contacts[j].normal)
            for contactPoint, contactNormal in zip(
                contactPoints[::2], contactNormals[::2]
            ):
                if np.linalg.norm(contactPoint) > 0.3:
                    dartScene.addArrow(
                        contactPoint,
                        contactPoint
                        - 0.08 * contactNormal / np.linalg.norm(contactNormal),
                        pointCloudColor,
                        radius=0.0018,
                    )
            contactPoints = [
                c.point for c in dartScene.world.getLastCollisionResult().getContacts()
            ]
            # contactPointCloudShape.setPoint(contactPoints)
            dartScene.showFrame()

            if saveOpt["savePlots"] and (step % 10 == 0) and step > 300:
                dartScene.saveFrame(
                    saveOpt["saveFolderPath"] + "/contactSim_" + str(step),
                    eye=camEye,
                    center=camCenter,
                    up=camUp,
                    width=width,
                    height=height,
                )
    print("Done.")
