import os
import sys
import argparse
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
from pytest import approx

try:
    sys.path.append(os.getcwd().replace("/tests", ""))
    from src.modelling.discretization import (
        determineNumSegments,
        # calcualteSegmentLength,
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

# previous version
# length = 0.2
# minBendingRadius = 0.01
# errorLengthTol = 0.03
# errorWidthTol = 0.003

# numSegments = determineNumSegments(
#     length, minBendingRadius, errorLengthTol, errorWidthTol
# )


# print("Suitable number of segments(secant approximation): {}".format(numSegments))
# print("Segment length (secant approximation): {}".format(length / numSegments))

# numSegments = determineNumSegments(
#     length, minBendingRadius, errorLengthTol, errorWidthTol, secantMode=False
# )

# print("Suitable number of segments(tangent approximation): {}".format(numSegments))
# print("Segment length (tangent approximation): {}".format(length / numSegments))

# new simplified version
length = 1
bending_radius = 0.013
max_tolerated_error = 0.01

numSegments = determineNumSegments(
    total_length=length,
    minimal_bending_radius=bending_radius,
    max_tolerated_error=max_tolerated_error,
)

print(numSegments)
