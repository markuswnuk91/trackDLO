import sys, os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from matplotlib.patches import Ellipse
sys.path.append(os.path.abspath('.'))
from src.visualization.colors import thesisColorPalettes
from src.visualization.plotUtils import get_tex_fonts

saveFig = True
savePath = "imgs/gmmExample/"
global colors 
color_1 = thesisColorPalettes["viridis"].to_rgba(0)[:3]
color_2 = thesisColorPalettes["viridis"].to_rgba(0.5)[:3]
color_3 = thesisColorPalettes["viridis"].to_rgba(1)[:3]
colors = [color_1, color_2, color_3] 
# point_color = [0.121,0.4666,0.705] # matplotlib default blue
point_color = [0.5,0.5,0.5] # gray
tex_fonts = get_tex_fonts(
        latexFontSize_in_pt = 20,
        latexFootNoteFontSize_in_pt = 16,
        )
plt.rcParams.update(tex_fonts)

# Function to plot Gaussians as ellipses
def plot_gaussian_ellipse(ax, mean, cov, color, alpha=0.3):
    v, w = np.linalg.eigh(cov)
    v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
    u = w[0] / np.linalg.norm(w[0])
    angle = np.arctan(u[1] / u[0])
    angle = np.degrees(angle)
    ell = Ellipse(mean, v[0], v[1], 180.0 + angle, color=color, alpha=alpha)
    ax.add_patch(ell)


# Generate synthetic data
np.random.seed(42)
n_samples = 150

cluster_means = [(-3, -3), (4, 4), (4, -2.5)]
gmm_init_means = np.array([[0, 0], [-1, 2], [1.5, 1.5]])
cluster_covs = [
    [[3, 0.5], [0.5, 3]],
    [[3, -2], [-2, 0.7]],
    [[2, 0.7], [0.7, 1.5]],
]

X = np.vstack(
    [
        np.random.multivariate_normal(mean, cov, n_samples)
        for mean, cov in zip(cluster_means, cluster_covs)
    ]
)

# Initialize GMM
gmm = GaussianMixture(
    n_components=3,
    covariance_type="full",
    random_state=0,
    init_params="random",
    max_iter=0,
)
gmm.means_init = gmm_init_means
gmm.fit(X)  # Initial fitting for initialization snapshot

# create figures and axes
figures = []
axes = []
for i in range(0,4):
    figures.append(plt.figure(figsize=(5,5)))
    axes.append(plt.subplot())
# Plot initialization snapshot

axes[0].scatter(X[:, 0], X[:, 1], s=10, alpha=0.5, color=point_color)
# axes[0].set_title("Initialization (random means, equal variance)")
axes[0].set_xlabel('$x$')
axes[0].set_ylabel('$y$')
axes[0].set_xticks([])
axes[0].set_yticks([])
# Plot initial random Gaussians with equal covariance
init_cov = np.array([[1, 0], [0, 1]])
axes[0].scatter(gmm.means_[:,0],gmm.means_[:,1], s=30, alpha=0.3, color=colors)
for mean, color in zip(gmm.means_, colors):
    plot_gaussian_ellipse(axes[0], mean, init_cov, color=color)
# Intermediate snapshots
for i, n_iter in enumerate([5, 15], start=1):
    gmm_iter = GaussianMixture(
        n_components=3,
        covariance_type="full",
        max_iter=n_iter,
        random_state=0,
        init_params="random",
    )
    gmm_iter.means_init = gmm_init_means
    gmm_iter.fit(X)
    axes[i].scatter(X[:, 0], X[:, 1], s=10, alpha=0.5,color=point_color)
    # axes[i].set_title(f"Iteration snapshot (iter={n_iter})")
    axes[i].set_xlabel('$x$')
    axes[i].set_ylabel('$y$')
    axes[i].set_xticks([])
    axes[i].set_yticks([])
    axes[i].scatter(gmm_iter.means_[:,0],gmm_iter.means_[:,1], s=30, alpha=0.3, color=colors)
    for mean, cov, color in zip(gmm_iter.means_, gmm_iter.covariances_, colors):
        plot_gaussian_ellipse(axes[i], mean, cov, color=color)

# Final fitting
gmm_final = GaussianMixture(
    n_components=3,
    covariance_type="full",
    max_iter=100,
    random_state=0,
    init_params="random",
)
gmm_final.means_init = gmm_init_means
gmm_final.fit(X)

# axes[3].scatter(X[:, 0], X[:, 1], s=10, alpha=0.5)
# Predict cluster labels using the final Gaussian Mixture Model
labels = gmm_final.predict(X)
# Plot the final fitted Gaussians with points colored by predicted labels

for idx, color in enumerate(colors):
    axes[3].scatter(X[labels == idx, 0], X[labels == idx, 1], s=10, alpha=0.3, color=color, label=f'Cluster {idx+1}')
axes[3].scatter(gmm_final.means_[:,0],gmm_final.means_[:,1], s=30, alpha=0.3, color=colors)
#axes[3].set_title("Final fitted Gaussians")
axes[3].set_xlabel('$x$')
axes[3].set_ylabel('$y$')
axes[3].set_xticks([])
axes[3].set_yticks([])
for mean, cov, color in zip(gmm_final.means_, gmm_final.covariances_, colors):
    plot_gaussian_ellipse(axes[3], mean, cov, color=color)

for i, ax in enumerate(axes):
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_aspect("equal")
    if saveFig:
            figures[i].savefig(
                savePath + "gmm_iteration_{}.pdf".format(i),
                bbox_inches="tight"
            )
plt.tight_layout()
plt.show()
