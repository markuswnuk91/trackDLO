import os
import sys
import numpy as np

# define curve shapes
def helixShape(s, widthScaling=1.0, heightScaling=1.0, frequency=1.0, offset=[0, 0, 0]):
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
