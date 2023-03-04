import os
import sys
import numpy as np
from numpy import genfromtxt
from pytest import approx
from sklearn import manifold, datasets
import matplotlib.pyplot as plt
from matplotlib import ticker
from numpy.random import RandomState

try:
    sys.path.append(os.getcwd().replace("/tests", ""))
    from src.dimreduction.mlle.mlle import Mlle
except:
    print("Imports for Test MLLE failed.")
    raise

vis = False  # enable for visualization


def plot_3d(points, points_color, title):
    x, y, z = points.T

    fig, ax = plt.subplots(
        figsize=(6, 6),
        facecolor="white",
        tight_layout=True,
        subplot_kw={"projection": "3d"},
    )
    fig.suptitle(title, size=16)
    col = ax.scatter(x, y, z, c=points_color, s=50, alpha=0.8)
    ax.view_init(azim=-60, elev=9)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.zaxis.set_major_locator(ticker.MultipleLocator(1))

    fig.colorbar(col, ax=ax, orientation="horizontal", shrink=0.6, aspect=60, pad=0.01)
    plt.show(block=False)


def plot_2d(points, points_color, title):
    fig, ax = plt.subplots(figsize=(3, 3), facecolor="white", constrained_layout=True)
    fig.suptitle(title, size=16)
    add_2d_scatter(ax, points, points_color)
    plt.show(block=False)


def add_2d_scatter(ax, points, points_color, title=None):
    x, y = points.T
    ax.scatter(x, y, c=points_color, s=50, alpha=0.8)
    ax.set_title(title)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())


def testMLLE():
    mlleProblem = Mlle(X, k, 2)
    Y = mlleProblem.solve()
    if vis:
        plot_2d(Y, color, "MLLE representation")
        plt.show(block=True)


# X = genfromtxt("tests/testdata/spr/Xinit.csv", delimiter=",")
# Y = genfromtxt("tests/testdata/spr/Y.csv", delimiter=",")
rng = RandomState(0)
n_points = 1500
X, color = datasets.make_s_curve(n_points, random_state=rng)

D = X.shape[1]
k = 12


if vis:
    plot_3d(X, color, "Original S-curve samples")

testMLLE()
