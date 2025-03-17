import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import LocallyLinearEmbedding
from mpl_toolkits.mplot3d import Axes3D

# Add your project root to Python path
sys.path.append(os.path.abspath('.'))
from src.visualization.plotUtils import get_tex_fonts
from src.visualization.plot3D import *
from src.visualization.plot2D import *

saveFig = True
savePath = "imgs/lleExample/"
tex_fonts = get_tex_fonts()
plt.rcParams.update(tex_fonts)

# Generate a twisted cable-like deformable linear object
n_points = 1500
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

# Perform Locally Linear Embedding
lle = LocallyLinearEmbedding(n_neighbors=15, n_components=2)
embedding = lle.fit_transform(points_3d)

# Plotting
# 3D plot
fig_1 = plt.figure()
ax1 = fig_1.add_subplot(projection='3d')
from src.visualization.plotUtils import get_tex_fonts
ax1.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c=colors, s=15)
# ax1.set_title('3D point cloud')
ax1.set_xlabel('$x$')
ax1.set_ylabel('$y$')
ax1.set_zlabel('$z$')

# 2D Embedding plot
fig_2 = plt.figure()
ax2 = fig_2.add_subplot()
ax2.scatter(embedding[:, 0], embedding[:, 1], c=colors, s=15)
# ax2.set_title('LLE Embedding Space')
ax2.set_xlabel('Embedded dimension $y_1$')
ax2.set_ylabel('Embedded dimension $y_2$')


# set background color as white
ax1.xaxis.pane.fill = False
ax1.yaxis.pane.fill = False
ax1.zaxis.pane.fill = False

plt.tight_layout()

if saveFig:
        fig_1.savefig(
            savePath + "lleExample_3D.pdf",
            bbox_inches="tight",
        )
        fig_2.savefig(
            savePath + "lleExample_embeddingSpace.pdf",
            bbox_inches="tight",
        )
plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.manifold import LocallyLinearEmbedding
# from mpl_toolkits.mplot3d import Axes3D

# # Helper function to generate datasets
# def generate_datasets(n_points=1500):
#     np.random.seed(42)

#     # 1. Planar Spiral with 3D Scattering
#     t_spiral = np.linspace(0, 4 * np.pi, n_points)
#     x_spiral = t_spiral * np.cos(t_spiral)
#     y_spiral = t_spiral * np.sin(t_spiral)
#     z_spiral = 0.3 * np.random.randn(n_points)
#     spiral = np.vstack((x_spiral, y_spiral, z_spiral)).T

#     # 2. Twisted cable-like DLO
#     t_cable = np.linspace(0, 4 * np.pi, n_points)
#     x_cable = np.cos(t_cable) + 0.1 * np.random.randn(n_points)
#     y_cable = np.sin(t_cable) + 0.1 * np.random.randn(n_points)
#     z_cable = t_cable / (2 * np.pi)
#     cable = np.vstack((x_cable, y_cable, z_cable)).T

#     # 3. Swiss Roll
#     t_swiss = 1.5 * np.pi * (1 + 2 * np.random.rand(n_points))
#     x_swiss = t_swiss * np.cos(t_swiss)
#     y_swiss = 21 * np.random.rand(n_points)
#     z_swiss = t_swiss * np.sin(t_swiss)
#     swiss = np.vstack((x_swiss, y_swiss, z_swiss)).T

#     return (spiral, t_spiral), (cable, t_cable), (swiss, t_swiss)

# # Generate datasets
# datasets = generate_datasets()
# titles = ['3D Planar Spiral with Scattering', 'Twisted Cable-like DLO', 'Swiss Roll']

# fig = plt.figure(figsize=(18, 12))

# for i, (data, t) in enumerate(datasets):
#     colors = plt.cm.viridis(t / max(t))

#     # Perform LLE
#     lle = LocallyLinearEmbedding(n_neighbors=15, n_components=2)
#     embedding = lle.fit_transform(data)

#     # 3D Plot
#     ax = fig.add_subplot(3, 2, 2*i + 1, projection='3d')
#     ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=colors, s=15)
#     ax.set_title(f'{titles[i]} (3D)')
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')

#     # 2D Embedding plot
#     ax = fig.add_subplot(3, 2, 2*i + 2)
#     ax.scatter(embedding[:, 0], embedding[:, 1], c=colors, s=15)
#     ax.set_title(f'{titles[i]} - LLE Embedding')
#     ax.set_xlabel('Component 1')
#     ax.set_ylabel('Component 2')

# plt.tight_layout()
# plt.show()