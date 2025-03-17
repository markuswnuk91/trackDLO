import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from matplotlib.patches import Ellipse


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

cluster_means = [(-4, 4), (4, 6), (3, 0)]
gmm_init_means = np.array([[-3, -3], [1, 2], [2, 2]])
cluster_covs = [
    [[1, 0.2], [0.2, 1]],
    [[0.5, -0.4], [-0.4, 0.5]],
    [[1.5, 0.5], [0.5, 1.0]],
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

# Plot initialization snapshot
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
axes[0].scatter(X[:, 0], X[:, 1], s=10, alpha=0.5)
axes[0].set_title("Initialization (random means, equal variance)")

# Plot initial random Gaussians with equal covariance
init_cov = np.array([[1, 0], [0, 1]])
colors = ["r", "g", "b"]
for mean, color in zip(gmm.means_, colors):
    plot_gaussian_ellipse(axes[0], mean, init_cov, color=color)

# Intermediate snapshots
for i, n_iter in enumerate([2, 5], start=1):
    gmm_iter = GaussianMixture(
        n_components=3,
        covariance_type="full",
        max_iter=n_iter,
        random_state=0,
        init_params="random",
    )
    gmm_iter.means_init = gmm_init_means
    gmm_iter.fit(X)
    axes[i].scatter(X[:, 0], X[:, 1], s=10, alpha=0.5)
    axes[i].set_title(f"Iteration snapshot (iter={n_iter})")
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

axes[3].scatter(X[:, 0], X[:, 1], s=10, alpha=0.5)
axes[3].set_title("Final fitted Gaussians")
for mean, cov, color in zip(gmm_final.means_, gmm_final.covariances_, colors):
    plot_gaussian_ellipse(axes[3], mean, cov, color=color)

for ax in axes:
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_aspect("equal")

plt.tight_layout()
plt.show()
