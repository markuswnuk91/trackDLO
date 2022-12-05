import os, sys
import numpy as np
import math
from warnings import warn
import numbers


def determineNumSegments(
    length,
    minBendingRadius,
    errorLengthTol,
    errorWidthTol,
    numSegmentsTol=None,
    secantMode=True,
):
    """
    Determine a suitable number of segments to discretize the continous shape of a DLO with a finite segment model.

    Attributes
    ----------
    minBendingRadius [float]:   the expected minimal bending radius of the DLO (given   by the inverse of the maximum curvature of the DLO) in meter

    length [float]: the length of the DLO im meter

    errorLengthCumTol [float]: maximum tolerated cumulated length error between the continous and discrete representation assuming a constant maximum curvature with the given minimal bending radius.

    errorWidthTol [float]: maximum tolerated width error between the continous and discrete representation assuming a constant maximum curvature with the given minimal bending radius.

    numSegmentsTol [int]: optional, maximum tolerated Number of Segments

    secantMode: optional, if the discretization should account for a secant-based approximation (joints are assumed to lie on the continous shape) or a tangent-based approximation (middle of segments are assumed to lie on the continous shape). Defaults to secant-based.


    Returns
    -------
    numSegments [int]: number of segments the DLO should be discretized into.
    """

    if length is not None and (not isinstance(length, numbers.Number) or length < 0):
        raise ValueError(
            "Expected a positive value for the DLO's length. Instead got: {}".format(
                length
            )
        )

    if minBendingRadius is not None and (
        not isinstance(minBendingRadius, numbers.Number) or minBendingRadius < 0
    ):
        raise ValueError(
            "Expected a positive value for the DLO's expected minimal bending radius. Instead got: {}".format(
                minBendingRadius
            )
        )
    elif 2 * minBendingRadius > length:
        raise ValueError(
            "Expected the DLO's length to be larger than the expected minimal bending radius: {}".format(
                minBendingRadius
            )
        )

    if errorLengthTol is not None and (
        not isinstance(errorLengthTol, numbers.Number) or errorLengthTol < 0
    ):
        raise ValueError(
            "Expected a positive value for the tolerated length error. Instead got: {}".format(
                errorLengthTol
            )
        )
    elif errorLengthTol >= length:
        raise ValueError(
            "Expected the tolerated length error to be smaller than the DLO's length."
        )

    if errorWidthTol is not None and (
        not isinstance(errorWidthTol, numbers.Number) or errorWidthTol < 0
    ):
        raise ValueError(
            "Expected a positive value for the tolerated length error. Instead got: {}".format(
                errorWidthTol
            )
        )

    if numSegmentsTol is not None and (
        not isinstance(numSegmentsTol, numbers.Number) or numSegmentsTol < 0
    ):
        raise ValueError(
            "Expected a positive value for the tolerated length error. Instead got: {}".format(
                numSegmentsTol
            )
        )
    elif isinstance(numSegmentsTol, numbers.Number) and not isinstance(
        numSegmentsTol, int
    ):
        warn(
            "Received a non-integer value for numSegmentsTol: {}. Casting to integer.".format(
                numSegmentsTol
            )
        )
        numSegmentsTol = int(numSegmentsTol)

    if secantMode is not isinstance(secantMode, bool):
        warn("Received a non-bool for mode. Defaulting to secant-based mode")
        secantMode = True

    numSegments = 1
    segmentLength = calcualteSegmentLength(numSegments, length)

    while segmentLength > minBendingRadius:
        segmentLength = calcualteSegmentLength(numSegments, length)
        numSegments += 1

    if secantMode:
        segmentLength = calcualteSegmentLength(numSegments, length)
        errorWidth = calculateWidthErrorSecant(minBendingRadius, segmentLength)
        while errorWidth > errorWidthTol:
            segmentLength = calcualteSegmentLength(numSegments, length)
            errorWidth = calculateWidthErrorSecant(minBendingRadius, segmentLength)
            numSegments += 1

        segmentLength = calcualteSegmentLength(numSegments, length)
        errorLength = calculateLengthErrorSecant(
            minBendingRadius, segmentLength, numSegments
        )
        while errorLength > errorLengthTol:
            segmentLength = calcualteSegmentLength(numSegments, length)
            errorLength = calculateLengthErrorSecant(
                minBendingRadius, segmentLength, numSegments
            )
            numSegments += 1
    else:
        segmentLength = calcualteSegmentLength(numSegments, length)
        errorWidth = calculateWidthErrorTangent(minBendingRadius, segmentLength)
        while errorWidth > errorWidthTol:
            segmentLength = calcualteSegmentLength(numSegments, length)
            errorWidth = calculateWidthErrorTangent(minBendingRadius, segmentLength)
            numSegments += 1

        segmentLength = calcualteSegmentLength(numSegments, length)
        errorLength = calculateLengthErrorTangent(minBendingRadius, segmentLength)
        while errorLength > errorLengthTol:
            segmentLength = calcualteSegmentLength(numSegments, length)
            segmenerrorLength = calculateLengthErrorTangent(
                minBendingRadius, segmentLength, numSegments
            )
            numSegments += 1

    return numSegments


def calcualteSegmentLength(numSegments, length):
    return length / numSegments


def calculateWidthErrorSecant(minBendingRadius, segmentLength):
    return minBendingRadius - np.sqrt(minBendingRadius**2 - (segmentLength / 2) ** 2)


def calculateLengthErrorSecant(minBendingRadius, segmentLength, numSegments):
    return numSegments * (
        2 * minBendingRadius * np.arcsin(segmentLength / (2 * minBendingRadius))
        - segmentLength
    )


def calculateWidthErrorTangent(minBendingRadius, segmentLength):
    raise NotImplementedError


def calculateLengthErrorTangent(minBendingRadius, segmentLength, numSegments):
    raise NotImplementedError
