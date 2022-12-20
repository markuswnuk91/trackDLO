import os
import sys
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
from pytest import approx

try:
    sys.path.append(os.getcwd().replace("/tests", ""))
    from src.modelling.wakamatsuModel import (
        WakamatsuModel,
    )
except:
    print("Imports for WakamatsuModel failed.")
    raise
vis = True  # enable for visualization


def testInit():
    testModel = WakamatsuModel()


def testFunctions():
    testModel = WakamatsuModel(**{"L": 2})
    STest = np.array([0, 0.5, 1, 2])
    assert testModel.evalPositions(STest, 100)[0, -1] == 0
    assert testModel.evalPositions(STest, 100)[1, -1] == 0.5
    assert testModel.evalPositions(STest, 100)[2, -1] == 1
    assert testModel.evalPositions(STest, 100)[3, -1] == 2


if __name__ == "__main__":
    testInit()
    testFunctions()
