import numpy as np
import matplotlib.pyplot as plt


def plot3DCurve(X, color=[1, 0, 0], block=False):
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.plot3D(X[:, 0], X[:, 1], X[:, 2], color=color)
    plt.show(block=block)
