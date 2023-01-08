import matplotlib.pyplot as plt
import numpy as np


def plotPointSet(
    X,
    ax,
    color=[0, 0, 1],
    alpha=1,
    label: str = None,
    axisLimX=[0, 1],
    axisLimY=[0, 1],
    axisLimZ=[0, 1],
    waitTime=0.001,
):
    if label is None:
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], color=color, alpha=alpha)
    else:
        ax.scatter(
            X[:, 0],
            X[:, 1],
            X[:, 2],
            color=color,
            label=label,
            alpha=alpha,
            edgecolor=None,
        )
    ax.set_xlim(axisLimX[0], axisLimX[1])
    ax.set_ylim(axisLimY[0], axisLimY[1])
    ax.set_zlim(axisLimZ[0], axisLimZ[1])
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$z$")
    if waitTime is not None or waitTime == -1:
        plt.draw()
        plt.pause(waitTime)
    elif waitTime is None:
        plt.show(block=False)
    elif waitTime == -1:
        plt.show(block=True)


def plotPointSets(
    X,
    Y,
    ax,
    axisLimX=[0, 1],
    axisLimY=[0, 1],
    axisLimZ=[0, 1],
    waitTime=0.001,
):
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], color="blue", label="Source")
    ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], color="red", label="Target")
    # plt.text(
    #     0.7,
    #     0.92,
    #     s="Wakamatsu Model Reconstruction",
    #     horizontalalignment="center",
    #     verticalalignment="center",
    #     transform=ax.transAxes,
    #     fontsize="x-large",
    # )
    ax.legend(loc="upper left", fontsize="x-large")
    ax.set_xlim(axisLimX[0], axisLimX[1])
    ax.set_ylim(axisLimY[0], axisLimY[1])
    ax.set_zlim(axisLimZ[0], axisLimZ[1])
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$z$")
    if waitTime is not None or waitTime == -1:
        plt.draw()
        plt.pause(waitTime)
    elif waitTime is None:
        plt.show(block=False)
    elif waitTime == -1:
        plt.show(block=True)


def plotPointSetAsLine(
    ax: plt.axis,
    X: np.array,
    label: str = None,
    color=[0, 0, 1],
    alpha=1,
    axisLimX=[0, 1],
    axisLimY=[0, 1],
    axisLimZ=[0, 1],
    waitTime=None,
):
    if label is None:
        ax.plot3D(X[:, 0], X[:, 1], X[:, 2], color=color, alpha=alpha)
    else:
        ax.plot3D(X[:, 0], X[:, 1], X[:, 2], color=color, label=label, alpha=alpha)
    # plt.text(
    #     0.7,
    #     0.92,
    #     s="Wakamatsu Model Reconstruction",
    #     horizontalalignment="center",
    #     verticalalignment="center",
    #     transform=ax.transAxes,
    #     fontsize="x-large",
    # )
    ax.legend(loc="upper left", fontsize="x-large")
    ax.set_xlim(axisLimX[0], axisLimX[1])
    ax.set_ylim(axisLimY[0], axisLimY[1])
    ax.set_zlim(axisLimZ[0], axisLimZ[1])
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$z$")
    if waitTime is not None or waitTime == -1:
        plt.draw()
        plt.pause(waitTime)
    elif waitTime == None:
        plt.show(block=False)
    elif waitTime == -1:
        plt.show(block=True)


def plotPointSetsAsLine(
    ax: plt.axis,
    X: np.array,
    Y: np.array,
    labelX: str = None,
    labelY: str = None,
    colorX=[0, 0, 1],
    colorY=[1, 0, 0],
    axisLimX=[0, 1],
    axisLimY=[0, 1],
    axisLimZ=[0, 1],
    waitTime=None,
):
    if labelX is None:
        ax.plot3D(X[:, 0], X[:, 1], X[:, 2], color=colorX)
    else:
        ax.plot3D(X[:, 0], X[:, 1], X[:, 2], color=colorX, label=labelX)

    if labelY is None:
        ax.plot3D(Y[:, 0], Y[:, 1], Y[:, 2], color=colorY)
    else:
        ax.plot3D(Y[:, 0], Y[:, 1], Y[:, 2], color=colorY, label=labelY)
    # plt.text(
    #     0.7,
    #     0.92,
    #     s="Wakamatsu Model Reconstruction",
    #     horizontalalignment="center",
    #     verticalalignment="center",
    #     transform=ax.transAxes,
    #     fontsize="x-large",
    # )
    ax.legend(loc="upper left", fontsize="x-large")
    ax.set_xlim(axisLimX[0], axisLimX[1])
    ax.set_ylim(axisLimY[0], axisLimY[1])
    ax.set_zlim(axisLimZ[0], axisLimZ[1])
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$z$")
    if waitTime is not None or waitTime == -1:
        plt.draw()
        plt.pause(waitTime)
    elif waitTime == None:
        plt.show(block=False)
    elif waitTime == -1:
        plt.show(block=True)
