import numpy as np


def readPointCloudFromPLY(path):
    points = []
    with open(path, "r") as f:
        file = f.readlines()

    for line in file[11:]:
        points.append(line.split(" ")[:6])

    return np.asarray(points, dtype=float)
