import sys
import os
import numpy as np

try:
    sys.path.append(os.getcwd().replace("/eval", ""))
    from src.evaluation.evaluation import (
        Evaluation,
    )
    from src.visualization.plot3D import *
    from scipy.spatial import distance_matrix
except:
    print("Imports for initialization evaluation script failed.")
    raise

dataSetPath = "data/darus_data_download/data/202230603_Configurations_mounted/20230603_143937_modelY/"

if __name__ == "__main__":
    # setup evaluation class
    dataSetName = dataSetPath.split("/")[-2]
    dataSetIdentifier = dataSetName
    relConfigFilePath = configFilePath = "/evalConfig.json"
    modelIdentifier = dataSetName.split("_")[-1]
    pathToConfigFile = os.path.dirname(os.path.abspath(__file__)) + relConfigFilePath
    eval = Evaluation(pathToConfigFile)

    #  load point cloud
    ptCloud = eval.getPointCloud(0, dataSetPath)
    # fig, ax = setupLatexPlot3D()
    # plotPointSet(ax=ax, X=ptCloud[0])
    # plt.show(block=True)
    # sample subset of points
    # Ensure N is not greater than the number of available points
    point_set = ptCloud[0]
    num_points = point_set.shape[1]
    N = 300
    N = min(N, num_points)
    random_indices = np.random.choice(num_points, size=N, replace=False)
    sampled_points_3D = point_set[:, random_indices]
    # determine set of distances between points in the cloud
    distances_3D = distance_matrix(sampled_points_3D, sampled_points_3D)

    # reproject sampled points
    sampled_points_2D = eval.reprojectFrom3DRobotBase(sampled_points_3D, dataSetPath)
    distances_2D = distance_matrix(sampled_points_2D, sampled_points_2D)

    distances_2D_vec = distances_2D.flatten()
    distances_3D_vec = distances_3D.flatten() * 1000  # scale to mm
    A = np.vstack([distances_2D_vec, np.ones(len(distances_2D_vec))]).T
    k, _ = np.linalg.lstsq(A, distances_3D_vec, rcond=None)[0]

    print("Scaling factor (milli meters/pixel):", k)
    print("Done.")

    # determine set of distances between points in the image

    # calculate approximate unit conversion factor
