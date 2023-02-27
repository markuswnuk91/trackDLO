import os
import sys
import argparse
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
from pytest import approx

try:
    sys.path.append(os.getcwd().replace("/tests", ""))
    from src.simulation.dlo import DeformableLinearObject
    from src.tracking.jspr.jspr import (
        JacobianBasedStructurePreservingRegistration,
        KinematicsModelDart,
    )
    from src.visualization.plot3D import plotPointSets, setupLatexPlot3D

    # from src.tracking.spr.spr import StructurePreservedRegistration
    # from src.tracking.cpd.cpd import CoherentPointDrift
except:
    print("Imports for Test JSPR failed.")
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


# def visualize(iteration, error, X, Y, ax):
#     plt.cla()
#     ax.scatter(X[:, 0], X[:, 1], color="blue", label="Source")
#     ax.scatter(Y[:, 0], Y[:, 1], color="red", label="Target")
#     plt.text(
#         0.7,
#         0.92,
#         "Iteration: {:d}, error{:.4f}".format(iteration, error),
#         horizontalalignment="center",
#         verticalalignment="center",
#         transform=ax.transAxes,
#         fontsize="x-large",
#     )
#     ax.legend(loc="upper left", fontsize="x-large")
#     plt.draw()
#     plt.pause(0.001)


def testJSPR():
    testDLO = DeformableLinearObject(13)
    kinematicModel = KinematicsModelDart(testDLO.skel.clone())
    qInit = kinematicModel.skel.getPositions()
    # print(kinematicModel.getPositions(qInit))
    # print(kinematicModel.getJacobian(qInit, 1))
    Y = kinematicModel.getPositions(0.3 * np.random.rand(qInit.shape[0]))
    Y = np.delete(Y, slice(3, 11), axis=0)
    reg = JacobianBasedStructurePreservingRegistration(
        **{
            "qInit": qInit,
            "Y": Y,
            "model": kinematicModel,
            "beta": 2,
            "lambdaAnnealing": 0.9,
            "max_iterations": 100,
            "damping": 0.1,
            "minDampingFactor": 0.1,
            "dampingAnnealing": 0.7,
            "stiffness": 100,
            "q0": np.zeros(qInit.shape[0]),
            "gravity": np.array([0, 0, -0.2]),
            "alpha": 30,
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
    testJSPR()
