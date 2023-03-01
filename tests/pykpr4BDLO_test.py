import os
import sys
import argparse
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
from pytest import approx
from scipy.spatial.transform import Rotation as R

try:
    sys.path.append(os.getcwd().replace("/tests", ""))
    from src.simulation.bdlo import bdloSpecification, BranchedDeformableLinearObject
    from src.tracking.kpr.kpr4BDLO import KinematicsPreservingRegistration4BDLO
    from src.tracking.kpr.kinematicsModel import KinematicsModelDart
    from src.visualization.plot3D import plotPointSets, setupLatexPlot3D
    from src.sensing.cameraModel import CameraModel
except:
    print("Imports for Test KPR4BDLO failed.")
    raise
vis = True  # enable for visualization


def setupVisualizationCallback(registration):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    return partial(
        visualizationCallback,
        fig,
        ax,
        registration,
        # savePath="/mnt/c/Users/ac129490/Documents/Dissertation/Software/trackdlo/imgs/continuousShapeReconstuction/helix_fail3/",
    )


def visualizationCallback(
    fig,
    ax,
    registration,
    savePath=None,
    fileName="img",
):
    if savePath is not None and type(savePath) is not str:
        raise ValueError("Error saving 3D plot. The given path should be a string.")

    if fileName is not None and type(fileName) is not str:
        raise ValueError("Error saving 3D plot. The given filename should be a string.")
    plt.cla()
    ax.set_xlim(-0.5, 1)
    ax.set_ylim(-0.5, 1)
    ax.set_zlim(-0.5, 1)
    plotPointSets(ax=ax, X=registration.T, Y=registration.Y, waitTime=0.5)
    if savePath is not None:
        fig.savefig(savePath + fileName + "_" + str(registration.iteration) + ".png")


def testKPR():
    testTopology = np.array([[0, 1, 0, 0], [1, 0, 1, 1], [0, 1, 0, 1], [0, 1, 1, 0]])
    testSpec = bdloSpecification(testTopology, defaultNumBodyNodes=30)
    testDLO = BranchedDeformableLinearObject(testSpec)
    testDLO.setBranchDof(0, 0, 0.3)
    kinematicModel = KinematicsModelDart(testDLO.skel.clone())
    B = []
    for i in range(0, testDLO.getNumBranches()):
        B.append(testDLO.getBranchBodyNodeIndices(i))
    qInit = kinematicModel.skel.getPositions()
    qTarget = qInit.copy()
    qTarget[0] = qTarget[0] + np.pi / 4
    qTarget[1] = qTarget[1] + np.pi / 4
    qTarget[2] = qTarget[2] + np.pi / 2
    qTarget[3] = qTarget[3] + 0.1
    qTarget[4] = qTarget[4] + 0.1
    qTarget[5] = qTarget[5] + 0.1
    # print(kinematicModel.getPositions(qInit))
    # print(kinematicModel.getJacobian(qInit, 1))
    Y = kinematicModel.getPositions(qTarget)
    # Y = np.delete(Y, slice(4, 7), axis=0)
    # camTransform = np.eye(4)
    # camTransform[:3, :3] = R.from_euler("ZYX", [180, 0, 235], degrees=True).as_matrix()
    # camTransform[:3, 3] = np.array([1, 1, 1])
    # camModel = CameraModel(
    #     camTransform=camTransform,
    #     X=Y,
    #     localTangents=np.vstack((np.diff(Y, axis=0), np.diff(Y, axis=0)[-1])),
    #     radius=0.05,
    # )
    # YCloud = camModel.calculatePointCloud()
    Dof = qInit.shape[0]
    stiffnessMatrix = 1 * np.eye(Dof)
    stiffnessMatrix[3:6, 3:6] = np.zeros((3, 3))
    reg = KinematicsPreservingRegistration4BDLO(
        **{
            "qInit": qInit,
            "q0": np.zeros(Dof),
            "Y": Y,
            "model": kinematicModel,
            "B": B,
            "max_iterations": 100,
            "damping": 3,
            "stiffnessMatrix": stiffnessMatrix,
            "gravity": np.array([0, 0, -1]),
            "mu": 0.0,
            "wCorrespondance": 10,
            "wStiffness": 500,
            "wGravity": 0,
            "minDampingFactor": 1,
            "dampingAnnealing": 0.9,
            "stiffnessAnnealing": 0.3,
            "gravitationalAnnealing": 1,
        }
    )
    if vis:
        callback = setupVisualizationCallback(reg)
        reg.register(callback)
        # plt.show()
    else:
        reg.register()

    print("Done")


if __name__ == "__main__":
    testKPR()
