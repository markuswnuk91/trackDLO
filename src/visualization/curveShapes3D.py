import os
import sys
import numpy as np

# define curve shapes
def helixShape(
    s, widthScaling=1.0, heightScaling=1.0, frequency=1.0, offset=np.array([0, 0, 0])
):
    """function that returns a set of vectors descibing the shape of a helix

    Args:
        s (np.array): Nx1 array of local coodinates for which the position vector should be returned
        offset (np.array): [x,y,z] offset vector
        width (float, optional): parameter to change the width of the helix
        frequency (flaot, optional): parameter to change the frequency of the helix

    Returns:
        np.array: NxD array of position vectors describing the points on the object corresponding to s
    """
    curveVecs = np.array(
        [
            widthScaling * np.cos(frequency * s * np.pi) + offset[0],
            widthScaling * np.sin(frequency * s * np.pi) + offset[1],
            heightScaling * s + offset[2],
        ]
    )
    return curveVecs.T


def gaussianShape(s, mu=0.5, width=1.0, height=1.0, offset=np.array([0, 0, 0])):
    """funcion that returns a set of vectors describing the shape of a gaussian

    Args:
        s (np.array): Nx1 array of local coordinates in [0,1] for which the position vectors should be returned
        sigma (float, optional): parameter to control the widht of the gaussian function. Defaults to 1.0.
        height (float, optional): parameter to control the height of the gaussian function. Defaults to 1.0.
        offset (_type_, optional): Offset vector. Defaults to np.array[0, 0, 0].

    Returns:
        np.array: NxD array of position vectors describing the points on the object corresponding to s
    """
    curveVecs = np.array(
        [
            s + offset[0],
            height * np.exp(-0.5 / width * ((s - mu) ** 2)) + offset[1],
            np.zeros(s.size) + offset[2],
        ]
    )
    return curveVecs.T
