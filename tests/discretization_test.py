import os
import sys
import argparse
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
from pytest import approx

try:
    sys.path.append(os.getcwd().replace("/tests", ""))
    from src.simulation.discretization import (
        determineNumSegments,
        calcualteSegmentLength,
    )
except:
    print("Imports for discretization failed.")
    raise
vis = True  # enable for visualization

# DLO parameters
length = 1
minBendingRadius = 0.12
errorLengthTol = 0.03
errorWidthTol = 0.005

numSegments = determineNumSegments(
    length, minBendingRadius, errorLengthTol, errorWidthTol
)

print("Suitable number of segments: {}".format(numSegments))
print("Segment length: {}".format(length / numSegments))
