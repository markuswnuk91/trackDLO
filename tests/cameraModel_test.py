# tests sensor model implementation
import os, sys
import dartpy as dart
import numpy as np
import math
import time as time
from pytest import approx
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

try:
    sys.path.append(os.getcwd().replace("/tests", ""))
    from src.modelling.finiteSegmentModel import FiniteSegmentModel
    from src.modelling.wakamatsuModel import (
        WakamatsuModel,
    )
    from src.sensing.cameraModel import CameraModel
    from src.visualization.plot3D import *
    from src.visualization.plotCoordinateSystems import plotCoordinateSystem
except:
    print("Imports for Sensor Model Tests failed.")
    raise

visualize = True


def testCameraModel():
    N = 10
    x0 = np.array([0.5, 0.5, 0])
    aPhi = 0 * np.ones(N)
    aTheta = 0 * np.ones(N)
    aPsi = 0 * np.ones(N)
    aPhi[2] = 2
    aTheta[0] = np.pi / 2
    # aTheta[1] = np.pi
    dloModel = WakamatsuModel(
        **{
            "L": 1,
            "aPhi": aPhi,
            "aTheta": aTheta,
            "aPsi": aPsi,
            "x0": x0,
        }
    )
    camTransform = np.eye(4)
    camTransform[:3, :3] = R.from_euler("ZYX", [180, 0, 235], degrees=True).as_matrix()
    camTransform[:3, 3] = np.array([1, 1, 1])
    sEval = np.linspace(0, dloModel.L, 100)

    camModel = CameraModel(
        camTransform=camTransform,
        X=dloModel.evalPositions(sEval),
        localTangents=dloModel.evalZeta(sEval).T,
        radius=0.01,
    )

    if visualize:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")

        plotCoordinateSystem(
            ax,
            camModel.camTransform,
            scale=0.1,
            arrowSize=10,
            offsetScale=1.1,
            originText=None,
            xText=None,
            yText=None,
            zText=None,
        )

        plotPointSet(
            camModel.X,
            ax,
            color=[0, 0, 0],
            alpha=1,
            label=None,
            size=20,
            waitTime=-1,
        )

        plotPointSet(
            camModel.calculatePointCloud(),
            ax,
            color=[1, 0, 0],
            alpha=1,
            label=None,
            size=20,
            waitTime=-1,
        )

        plt.show(block=True)


if __name__ == "__main__":
    testCameraModel()
