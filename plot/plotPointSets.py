import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Add your project root to Python path
sys.path.append(os.path.abspath('.'))
from src.visualization.plotUtils import get_tex_fonts
from src.visualization.plot3D import *
from src.visualization.plot2D import *

saveFig = True
savePath = "imgs/pointSets/"
tex_fonts = get_tex_fonts(
        latexFontSize_in_pt = 20,
        latexFootNoteFontSize_in_pt = 16,
        )

plt.rcParams.update(tex_fonts)
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

# Generate a twisted cable-like deformable linear object
n_points = 1000
np.random.seed(42)
t = np.linspace(0, 4 * np.pi, n_points)
x = np.cos(t)
y = np.sin(t)
z = t / (2 * np.pi)

# Introduce random deformations perpendicular to the main axis
x_noise = 0.1 * np.random.randn(n_points)
y_noise = 0.1 * np.random.randn(n_points)
x += x_noise
y += y_noise

points_3d = np.vstack((x, y, z)).T
colors = plt.cm.viridis(t / max(t))

# Background noise: unrelated points
n_noise = 700
x_bg = np.random.uniform(-1.4, 1.4, n_noise)
y_bg = np.random.uniform(-1.4, 1.4, n_noise)
z_bg = np.random.uniform(0, 0.8*max(z), n_noise)
noise_points = np.vstack((x_bg, y_bg, z_bg)).T

# Subsets for the variations
ordered = points_3d.copy()
ordered_colors = plt.cm.viridis(t / max(t))  # shape: (N, 4)

unordered = points_3d.copy()
unordered_colors = ordered_colors.copy()
np.random.seed(42)  # for reproducibility
np.random.shuffle(unordered_colors)


# Noisy: add more severe noise
noisy = np.vstack((points_3d, noise_points))
noisy_colors = np.vstack((ordered_colors, plt.cm.viridis(np.random.rand(len(noise_points)))))
occluded_ids = plt.cm.viridis(t / max(t))


# Segmented: artificially assign 3 segments
segmented = np.vstack((points_3d, noise_points))
segmented_colors = np.vstack((ordered_colors, plt.cm.Greys(0.35*np.ones(len(noise_points)))))



# Plotting all four
figs = []
titles = ['Unordered', 'Ordered', 'Noisy', 'Segmented']
datasets = [unordered, ordered, noisy, segmented]
colorsets = [
    unordered_colors,                             # Unordered: suffeled
    ordered_colors,         # Ordered: linear gradient color
    noisy_colors,                       # Noisy
    segmented_colors                              # Segmented
]

for i, (data, color, title) in enumerate(zip(datasets, colorsets, titles)):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=color, s=10)
    # ax.set_title(title)
    # ax.set_xlabel('$x$')
    # ax.set_ylabel('$y$')
    # ax.set_zlabel('$z$')
    print(ax.get_xlim())
    print(ax.get_ylim())
    print(ax.get_zlim())
    ax.set_xlim([-2.17, 2.16])
    ax.set_ylim([-2.19, 2.19])
    ax.set_zlim([-0.1, 2.1])
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.dist = 6
    plt.axis("off")
    figs.append(fig)
    if saveFig:
            fig.savefig(
                savePath + "pointSet_{}.pdf".format(title),
                bbox_inches="tight",
                pad_inches=-0.25
            )
plt.show()