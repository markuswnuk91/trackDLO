import os
import sys
import numpy as np
from numpy import genfromtxt
from pytest import approx

try:
    sys.path.append(os.getcwd().replace("/tests", ""))
    from src.tracking.spr.spr import mlle
except:
    print("Imports for Test MLLE failed.")
    raise


def testMLLE():
    mlle(X, k, D)


X = genfromtxt("tests/testdata/spr/Xinit.csv", delimiter=",")
D = X.shape[1]
Y = genfromtxt("tests/testdata/spr/Y.csv", delimiter=",")
k = 7
testMLLE()
