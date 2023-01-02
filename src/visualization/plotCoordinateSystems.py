import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)


def visualizeCoordinateSystem(
    ax,
    T,
    scale=1,
    arrowSize=10,
    offsetScale=1.1,
    originText=None,
    xText=None,
    yText=None,
    zText=None,
):
    # Here we create the arrows:
    arrow_prop_dict = dict(
        mutation_scale=arrowSize, arrowstyle="-|>", shrinkA=0, shrinkB=0
    )

    origin = T[:3, 3]
    X = scale * T[:3, 0]
    Y = scale * T[:3, 1]
    Z = scale * T[:3, 2]

    XArrow = Arrow3D(
        [origin[0], origin[0] + X[0]],
        [origin[1], origin[1] + X[1]],
        [origin[2], origin[2] + X[2]],
        **arrow_prop_dict,
        color="r"
    )
    ax.add_artist(XArrow)

    YArrow = Arrow3D(
        [origin[0], origin[0] + Y[0]],
        [origin[1], origin[1] + Y[1]],
        [origin[2], origin[2] + Y[2]],
        **arrow_prop_dict,
        color="b"
    )
    ax.add_artist(YArrow)

    ZArrow = Arrow3D(
        [origin[0], origin[0] + Z[0]],
        [origin[1], origin[1] + Z[1]],
        [origin[2], origin[2] + Z[2]],
        **arrow_prop_dict,
        color="g"
    )
    ax.add_artist(ZArrow)

    if originText is not None:
        offsetPos = origin - (offsetScale - 1) * (X + Y + Z)
        ax.text(
            (offsetPos[0]),
            (offsetPos[1]),
            (offsetPos[2]),
            "$" + str(originText) + "$",
        )
    if xText is not None:
        ax.text(
            (origin[0] + X[0]) * offsetScale,
            (origin[1] + X[1]) * offsetScale,
            (origin[2] + X[2]) * offsetScale,
            r"$x$",
        )
    if yText is not None:
        ax.text(
            (origin[0] + Y[0]) * offsetScale,
            (origin[1] + Y[1]) * offsetScale,
            (origin[2] + Y[2]) * offsetScale,
            r"$y$",
        )
    if zText is not None:
        ax.text(
            (origin[0] + Z[0]) * offsetScale,
            (origin[1] + Z[1]) * offsetScale,
            (origin[2] + Z[2]) * offsetScale,
            r"$z$",
        )
