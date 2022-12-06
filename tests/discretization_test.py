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
# length = 0.5
# minBendingRadius = 0.12
# errorLengthTol = 0.03
# errorWidthTol = 0.01


length = 0.2
minBendingRadius = 0.08
errorLengthTol = 0.01
errorWidthTol = 0.01

numSegments = determineNumSegments(
    length, minBendingRadius, errorLengthTol, errorWidthTol
)


print("Suitable number of segments(secant approximation): {}".format(numSegments))
print("Segment length (secant approximation): {}".format(length / numSegments))

numSegments = determineNumSegments(
    length, minBendingRadius, errorLengthTol, errorWidthTol, secantMode=False
)

print("Suitable number of segments(tangent approximation): {}".format(numSegments))
print("Segment length (tangent approximation): {}".format(length / numSegments))
