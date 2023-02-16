import os
import sys
import numpy as np
import matplotlib.pyplot as plt

try:
    sys.path.append(os.getcwd().replace("/tests", ""))
    from src.visualization.plotCoordinateSystems import (
        plotCoordinateSystem,
    )
except:
    print("Imports for plot Test failed.")
    raise


def test_plotCoordinateSystem():
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    plotCoordinateSystem(
        ax=ax,
        T=np.eye(4),
        scale=0.1,
        arrowSize=10,
        offsetScale=1.01,
        originText="o",
        xText="x",
        yText="y",
        zText="z",
    )
    plt.show(block=True)


if __name__ == "__main__":
    test_plotCoordinateSystem()
