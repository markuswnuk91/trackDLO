import numpy as np
import matplotlib.pyplot as plt


def plot3DCurve(X, color=[1, 0, 0], block=False, savePath=None, fileName="img.png"):
    if savePath is not None and type(savePath) is not str:
        raise ValueError("Error saving 3D plot. The given path should be a string.")

    if fileName is not None and type(fileName) is not str:
        raise ValueError("Error saving 3D plot. The given filename should be a string.")
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.plot3D(X[:, 0], X[:, 1], X[:, 2], color=color)
    plt.show(block=block)
    if savePath is not None:
        plt.savefig(savePath + fileName)
