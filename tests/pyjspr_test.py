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

    # from src.tracking.spr.spr import StructurePreservedRegistration
    # from src.tracking.cpd.cpd import CoherentPointDrift
except:
    print("Imports for Test JSPR failed.")
    raise
vis = True  # enable for visualization


def visualizationCallback(
    fig,
    ax,
    discreteModel,
    axisLimX=[0, 1],
    axisLimY=[0, 1],
    axisLimZ=[0, 1],
    savePath=None,
    fileName="img",
):
    if savePath is not None and type(savePath) is not str:
        raise ValueError("Error saving 3D plot. The given path should be a string.")

    if fileName is not None and type(fileName) is not str:
        raise ValueError("Error saving 3D plot. The given filename should be a string.")
    plt.cla()
    plotPointSets(
        X=discreteModel.X,
        Y=discreteModel.Y,
        ax=ax,
        axisLimX=axisLimX,
        axisLimY=axisLimY,
        axisLimZ=axisLimZ,
    )
    if savePath is not None:
        fig.savefig(savePath + fileName + "_" + str(discreteModel.iter) + ".png")


def visualize(iteration, error, X, Y, ax):
    plt.cla()
    ax.scatter(X[:, 0], X[:, 1], color="blue", label="Source")
    ax.scatter(Y[:, 0], Y[:, 1], color="red", label="Target")
    plt.text(
        0.7,
        0.92,
        "Iteration: {:d}, error{:.4f}".format(iteration, error),
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax.transAxes,
        fontsize="x-large",
    )
    ax.legend(loc="upper left", fontsize="x-large")
    plt.draw()
    plt.pause(0.001)


def testJSPR():
    testDLO = DeformableLinearObject(13)
    kinematicModel = KinematicsModelDart(testDLO.skel.clone())
    qInit = kinematicModel.skel.getPositions()
    print(kinematicModel.getPositions(qInit))
    print(kinematicModel.getJacobian(qInit, 1))
    Y = kinematicModel.getPositions(0.3 * np.random.rand(qInit.shape[0]))

    reg = JacobianBasedStructurePreservingRegistration(
        **{
            "qInit": qInit,
            "Y": Y,
            "model": kinematicModel,
            "lambdaFactor": 10,
            "beta": 0.1,
        }
    )
    if vis:
        fig = plt.figure()
        fig.add_axes([0, 0, 1, 1])
        callback = partial(visualize, ax=fig.axes[0])
        reg.register(callback)
        # plt.show()
    else:
        reg.register()


if __name__ == "__main__":
    testJSPR()
