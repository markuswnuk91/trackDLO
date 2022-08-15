import os
import sys
import argparse
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
from pytest import approx

try:
    sys.path.append(os.getcwd().replace("/tests", ""))
    from src.tracking.cpd.cpd import CoherentPointDrift
except:
    print("Imports for CPD failed.")
    raise
vis = False # enable for visualization

def visualize(iteration, error, X, Y, ax):
    plt.cla()
    ax.scatter(X[:, 0],  X[:, 1], color='red', label='Target')
    ax.scatter(Y[:, 0],  Y[:, 1], color='blue', label='Source')
    plt.text(0.87, 0.92, 'Iteration: {:d}'.format(
        iteration), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize='x-large')
    ax.legend(loc='upper left', fontsize='x-large')
    plt.draw()
    plt.pause(0.001)


def runCPD():
    X = np.loadtxt('data/pycpd_testdata/fish_source.txt')
    Y = np.loadtxt('data/pycpd_testdata/fish_target.txt')

    reg = CoherentPointDrift(**{'X': X, 'Y': Y})
    if vis:
        fig = plt.figure()
        fig.add_axes([0, 0, 1, 1])
        callback = partial(visualize, ax=fig.axes[0])
        reg.register(callback)
        plt.show()
    else:
        reg.register()
    T_ref = np.loadtxt('data/pycpd_testdata/fish_deformable_2D_result_Targets.txt')
    return(reg.T)


def testCPD():
    T_test = runCPD()
    T_ref = np.loadtxt('data/pycpd_testdata/fish_deformable_2D_result_Targets.txt')
    assert T_test == approx(T_ref)




if __name__ == '__main__':
    runCPD()

