import matplotlib.pyplot as plt
import numpy as np


def set_size(width, height=None, fraction=1, subplots=(1, 1)):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == "thesis":
        width_pt = 426.79135
    elif width == "beamer":
        width_pt = 307.28987
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    if height is None:
        fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])
    else:
        fig_height_in = height * inches_per_pt
    return (fig_width_in, fig_height_in)


def set_axes_equal3D(ax):
    """Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def set_axes_equal(ax):
    if ax.name == "3d":
        set_axes_equal3D(ax)
    else:
        ax.set_aspect("equal", adjustable="box")


def get_axis_limits(points):
    x_max = np.max(points[:, 0])
    x_min = np.min(points[:, 0])
    y_max = np.max(points[:, 1])
    y_min = np.min(points[:, 1])
    z_max = np.max(points[:, 2])
    z_min = np.min(points[:, 2])
    return {
        "xMin": x_min,
        "xMax": x_max,
        "yMin": y_min,
        "yMax": y_max,
        "zMin": z_min,
        "zMax": z_max,
    }


def center_point_cloud(points):
    centroid = np.mean(points, axis=0)
    return points - centroid


def scale_axes_to_fit(ax, points, zoom=1):
    ax.set_position(
        [
            0,
            0,
            1,
            1,
        ]
    )
    axis_limits = get_axis_limits(points)
    ax_range = np.max(
        np.array(
            (
                axis_limits["xMax"] - axis_limits["xMin"],
                axis_limits["yMax"] - axis_limits["yMin"],
                axis_limits["zMax"] - axis_limits["zMin"],
            )
        )
    )
    centroid = np.mean(points, axis=0)
    ax_offset = 2 * zoom
    ax.set_xlim(centroid[0] - ax_range / ax_offset, centroid[0] + ax_range / ax_offset)
    ax.set_ylim(centroid[1] - ax_range / ax_offset, centroid[1] + ax_range / ax_offset)
    ax.set_zlim(centroid[2] - ax_range / ax_offset, centroid[2] + ax_range / ax_offset)
    return ax
