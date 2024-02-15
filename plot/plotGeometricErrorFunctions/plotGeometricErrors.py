import sys
import os
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np

try:
    sys.path.append(os.getcwd().replace("/plot", ""))
    from src.modelling.discretization import calc_geometric_error
    from src.visualization.colors import *
    from src.visualization.plotUtils import *
except:
    print("Imports for plotting geometric errors failed.")
    raise

colorPalette = thesisColorPalettes["viridis"]
e_max = 0.005

if __name__ == "__main__":
    set_text_to_latex_font()

    bending_radii = np.linspace(0.01, 0.1, 10)
    N_segments = np.arange(1, 11, 1)
    for i, bending_radius in enumerate(bending_radii):
        color_s = i / len(bending_radii)
        color = colorPalette.to_rgba(color_s)[:3]
        geometric_errors = []
        for n_segments in N_segments:
            geometric_error = calc_geometric_error(bending_radius, n_segments)
            geometric_errors.append(geometric_error)
        geometric_errors = np.array(geometric_errors)
        plt.plot(N_segments, geometric_errors, color=color)
    ax = plt.gca()
    ax.set_xlim([1, N_segments[-1]])
    ax.set_ylim([0, 0.05])

    # labels
    ax.set_xlabel("number of segments")
    ax.set_ylabel("geometric error in m")

    plt.axhline(y=e_max, color="r", linestyle="--")
    # plt.axvline(x=3, ymax=0.1, ymin=0, color="r", linestyle="--")
    trans = transforms.blended_transform_factory(
        ax.get_yticklabels()[0].get_transform(), ax.transData
    )
    ax.text(
        0,
        e_max,
        r"$e_{max}$",
        color="red",
        transform=trans,
        ha="right",
        va="center",
    )
    lowerLim = 1 / bending_radii[0]
    upperLim = 1 / bending_radii[-1]
    norm = matplotlib.colors.Normalize(vmin=lowerLim, vmax=upperLim)  # Normalizer
    sm = plt.cm.ScalarMappable(cmap=matplotlib.colormaps["viridis"], norm=norm)
    ticks = 1 / bending_radii
    cbar = plt.colorbar(
        colorPalette,
        # ticks=ticks,
        location="right",
        # anchor=(-0.3, 0.5),
        # shrink=0.5,
    )
    cbar.ax.set_ylabel(
        "curvature",
        rotation=270,
        labelpad=0,
        # fontsize=latexFootNoteFontSize_in_pt,
    )
    # cbar.ax.yaxis.set_major_formatter(tick.FormatStrFormatter("%.1f"))
    # cbar.ax.set_ylabel(
    #     "torsion",
    #     rotation=270,
    #     labelpad=0,
    #     fontsize=latexFootNoteFontSize_in_pt,
    # )
    # cbar.ax.set_yticklabels(["low", "high"], fontsize=latexFootNoteFontSize_in_pt

    plt.show(block=True)
