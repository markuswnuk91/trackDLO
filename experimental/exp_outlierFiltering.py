import os, sys
import numpy as np
import random
from functools import partial
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor

try:
    sys.path.append(os.getcwd().replace("/experimental", ""))
except:
    print("Imports for SOM Test failed.")
    raise


dataPath = "data/experimental/outlier_filtering/somTestData.txt"
reducedPoints = np.loadtxt(dataPath)
# filter points
lofFilter = LocalOutlierFactor(n_neighbors=10, contamination=0.1)
result = lofFilter.fit_predict(reducedPoints)
negOutlierScore = lofFilter.negative_outlier_factor_
print(result)
print(negOutlierScore)
# visualization
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.set_xlim(0.2, 0.8)
ax.set_ylim(-0.3, 0.3)
ax.set_zlim(0, 0.6)
for i, point in enumerate(reducedPoints):
    (negOutlierScore.max() - negOutlierScore[i]) / (
        negOutlierScore.max() - negOutlierScore.min()
    )
    if result[i] == 1:
        color = np.array([0, 0, 1])
    else:
        color = np.array([1, 0, 0])
    ax.scatter(point[0], point[1], point[2], s=2 * i, color=color, alpha=0.2)
plt.show(block=True)
# plotPointSets(
#     ax=ax,
#     X=reducedPoints[np.where(result == 1), :],
#     Y=testCloud,
#     ySize=5,
#     xSize=10,
#     # yMarkerStyle=".",
#     yAlpha=0.01,
#     waitTime=None,
# )
