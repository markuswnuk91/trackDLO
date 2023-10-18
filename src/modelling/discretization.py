import os, sys
import numpy as np
import math
from warnings import warn
import numbers
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# def determineNumSegments(
#     length,
#     minBendingRadius,
#     errorLengthTol,
#     errorWidthTol,
#     numSegmentsTol=None,
#     secantMode=True,
# ):
#     """
#     Determine a suitable number of segments to discretize the continous shape of a DLO with a finite segment model.

#     Attributes
#     ----------
#     minBendingRadius [float]:   the expected minimal bending radius of the DLO (given   by the inverse of the maximum curvature of the DLO) in meter

#     length [float]: the length of the DLO im meter

#     errorLengthCumTol [float]: maximum tolerated cumulated length error between the continous and discrete representation assuming a constant maximum curvature with the given minimal bending radius.

#     errorWidthTol [float]: maximum tolerated width error between the continous and discrete representation assuming a constant maximum curvature with the given minimal bending radius.

#     numSegmentsTol [int]: optional, maximum tolerated Number of Segments

#     secantMode: optional, if the discretization should account for a secant-based approximation (joints are assumed to lie on the continous shape) or a tangent-based approximation (middle of segments are assumed to lie on the continous shape). Defaults to secant-based.


#     Returns
#     -------
#     numSegments [int]: number of segments the DLO should be discretized into.
#     """

#     if length is not None and (not isinstance(length, numbers.Number) or length < 0):
#         raise ValueError(
#             "Expected a positive value for the DLO's length. Instead got: {}".format(
#                 length
#             )
#         )

#     if minBendingRadius is not None and (
#         not isinstance(minBendingRadius, numbers.Number) or minBendingRadius < 0
#     ):
#         raise ValueError(
#             "Expected a positive value for the DLO's expected minimal bending radius. Instead got: {}".format(
#                 minBendingRadius
#             )
#         )
#     elif 2 * minBendingRadius > length:
#         raise ValueError(
#             "Expected the DLO's length to be larger than the diameter of the osculating circle of the expected maximum curvature: {}".format(
#                 2 * minBendingRadius
#             )
#         )

#     if errorLengthTol is not None and (
#         not isinstance(errorLengthTol, numbers.Number) or errorLengthTol < 0
#     ):
#         raise ValueError(
#             "Expected a positive value for the tolerated length error. Instead got: {}".format(
#                 errorLengthTol
#             )
#         )
#     elif errorLengthTol >= length:
#         raise ValueError(
#             "Expected the tolerated length error to be smaller than the DLO's length."
#         )

#     if errorWidthTol is not None and (
#         not isinstance(errorWidthTol, numbers.Number) or errorWidthTol < 0
#     ):
#         raise ValueError(
#             "Expected a positive value for the tolerated length error. Instead got: {}".format(
#                 errorWidthTol
#             )
#         )
#     elif errorWidthTol >= minBendingRadius:
#         raise ValueError(
#             "Expected the tolerated width error to be smaller than the minimal bending radius."
#         )

#     if numSegmentsTol is not None and (
#         not isinstance(numSegmentsTol, numbers.Number) or numSegmentsTol < 0
#     ):
#         raise ValueError(
#             "Expected a positive value for the tolerated length error. Instead got: {}".format(
#                 numSegmentsTol
#             )
#         )
#     elif isinstance(numSegmentsTol, numbers.Number) and not isinstance(
#         numSegmentsTol, int
#     ):
#         warn(
#             "Received a non-integer value for numSegmentsTol: {}. Casting to integer.".format(
#                 numSegmentsTol
#             )
#         )
#         numSegmentsTol = int(numSegmentsTol)

#     if not isinstance(secantMode, bool):
#         warn("Received a non-bool for mode. Defaulting to secant-based mode")
#         secantMode = True

#     numSegments = 1
#     segmentLength = calcualteSegmentLength(numSegments, length)

#     while segmentLength > 2 * minBendingRadius:
#         segmentLength = calcualteSegmentLength(numSegments, length)
#         numSegments += 1

#     if secantMode:
#         segmentLength = calcualteSegmentLength(numSegments, length)
#         errorWidth = calculateWidthErrorSecant(minBendingRadius, segmentLength)
#         while errorWidth > errorWidthTol:
#             segmentLength = calcualteSegmentLength(numSegments, length)
#             errorWidth = calculateWidthErrorSecant(minBendingRadius, segmentLength)
#             numSegments += 1

#         segmentLength = calcualteSegmentLength(numSegments, length)
#         errorLength = calculateLengthErrorSecant(
#             minBendingRadius, segmentLength, numSegments
#         )
#         while errorLength > errorLengthTol:
#             segmentLength = calcualteSegmentLength(numSegments, length)
#             errorLength = calculateLengthErrorSecant(
#                 minBendingRadius, segmentLength, numSegments
#             )
#             numSegments += 1
#     else:
#         segmentLength = calcualteSegmentLength(numSegments, length)
#         errorWidth = calculateWidthErrorTangent(minBendingRadius, segmentLength)
#         while errorWidth > errorWidthTol:
#             segmentLength = calcualteSegmentLength(numSegments, length)
#             errorWidth = calculateWidthErrorTangent(minBendingRadius, segmentLength)
#             numSegments += 1

#         segmentLength = calcualteSegmentLength(numSegments, length)
#         errorLength = calculateLengthErrorTangent(
#             minBendingRadius, segmentLength, numSegments
#         )
#         while errorLength > errorLengthTol:
#             segmentLength = calcualteSegmentLength(numSegments, length)
#             errorLength = calculateLengthErrorTangent(
#                 minBendingRadius, segmentLength, numSegments
#             )
#             numSegments += 1

#     return numSegments


# def calcualteSegmentLength(numSegments, length):
#     return length / numSegments


# def calculateWidthErrorSecant(minBendingRadius, segmentLength):
#     return minBendingRadius - np.sqrt(minBendingRadius**2 - (segmentLength / 2) ** 2)


# def calculateLengthErrorSecant(minBendingRadius, segmentLength, numSegments):
#     return numSegments * (
#         2 * minBendingRadius * np.arcsin(segmentLength / (2 * minBendingRadius))
#         - segmentLength
#     )


# def calculateWidthErrorTangent(minBendingRadius, segmentLength):
#     return np.sqrt(minBendingRadius**2 + (segmentLength / 2) ** 2) - minBendingRadius


# def calculateLengthErrorTangent(minBendingRadius, segmentLength, numSegments):
#     alpha = np.arctan(minBendingRadius / (segmentLength / 2))
#     s = segmentLength * np.sin(alpha)
#     eLength = segmentLength - (
#         2 * minBendingRadius * np.arcsin(s / (2 * minBendingRadius))
#     )
#     return numSegments * eLength


def piecewise_linear_approximation(r, n_segments, s):
    deltpa_phis = np.linspace(0, np.pi, n_segments + 1)
    segment_index = int((n_segments) * s) + 1

    # find the corresponding s_i, and s_j
    s_i = 0
    s_j = 0
    s_increment = 1 / n_segments
    while True:
        if s_i <= s and s_i + s_increment >= s:
            break
        else:
            s_i += s_increment
    s_j = s_i + s_increment
    # get the points on the circle
    x_i, y_i = circular_line(r, s=s_i)
    x_j, y_j = circular_line(r, s=s_j)
    # interpolate with linear segments
    x = x_i + (s - s_i) / (s_j - s_i) * (x_j - x_i)
    y = y_i + (s - s_i) / (s_j - s_i) * (y_j - y_i)
    return x, y


def circular_line(r, s):
    x = r * np.cos(np.pi * (1 - s))
    y = r * np.sin(np.pi * (1 - s))
    return x, y


def determineNumSegments(
    total_length,
    minimal_bending_radius,
    max_tolerated_error,
):
    n_segments = 1
    l_circle = np.pi * minimal_bending_radius
    l_segments = np.pi * minimal_bending_radius / n_segments
    s_space = np.linspace(0, 1, 100)

    all_errors = []
    n_iterations = 20
    for iteration, n_segments in enumerate(range(1, n_iterations)):
        errors = []
        P_piecewise = []
        P_circle = []
        for s in s_space:
            p_polygonial = np.array(
                piecewise_linear_approximation(
                    r=minimal_bending_radius, n_segments=n_segments, s=s
                )
            )
            p_circle = np.array(circular_line(r=minimal_bending_radius, s=s))
            P_piecewise.append(p_polygonial)
            P_circle.append(p_circle)
            errors.append(np.linalg.norm(p_polygonial - p_circle))
        # plt.plot(np.array(P_circle)[:, 0], np.array(P_circle)[:, 1], color="red")
        # plt.plot(np.array(P_piecewise)[:, 0], np.array(P_piecewise)[:, 1], color="blue")
        # plt.close("all")
        all_errors.append(errors)
        print("Iteration {}/{}".format(iteration + 1, n_iterations - 1))

    # determine max errors
    max_errors = []

    for errors in all_errors:
        max_error = np.max(errors)
        max_errors.append(max_error)
    n = np.array(list(range(1, len(max_errors) + 1)))
    plt.plot(n, max_errors, color="red")

    # fit a exponential function
    def func(x, a, b, c):
        return a * np.exp(-b * x) + c

    popt, pcov = curve_fit(func, n, max_errors)
    fit_result = np.polyfit(n, np.log(max_errors), 1)
    # plt.plot(n, func(n, *popt), color="blue")
    # invert the fitted function
    a, b, c = popt[0:3]
    reversed_fit = lambda x: -1 / b * np.log((x - c) / a)
    # get the number of segments form the maximum tolerated geometric error.
    n_segments = reversed_fit(max_tolerated_error)
    segment_length = l_circle / n_segments
    total_num_segments = np.ceil(total_length / segment_length)
    return total_num_segments
