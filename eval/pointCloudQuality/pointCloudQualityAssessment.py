import sys,os 
import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
try:
    sys.path.append(os.getcwd().replace("/eval", ""))
    from src.evaluation.evaluation import Evaluation

    # visualization
    from src.visualization.plot3D import *
except:
    print("Imports for script failed.")
    raise

dataSetPaths = [
    "data/darus_data_download/data/202230603_Configurations_mounted/20230603_143937_modelY/",
    "data/darus_data_download/data/20230807_Configurations_mounted/20230807_150735_partial/",
    "data/darus_data_download/data/202230603_Configurations_mounted/20230603_140143_arena/",
    "data/darus_data_download/data/20230516_Configurations_labeled/20230516_112207_YShape/",
    "data/darus_data_download/data/20230516_Configurations_labeled/20230516_113957_Partial/",  # finished 10.09.2023
    "data/darus_data_download/data/20230516_Configurations_labeled/20230516_115857_arena/", # finished 12.09.2023
]
global eval 
pathToConfigFile = os.path.dirname(os.path.abspath(__file__)) + "/evalConfig.json"
eval = Evaluation(pathToConfigFile)
vis=False
report_single=False

def load_point_cloud(frame,dataSetPath):
    pointCloud = eval.getPointCloud(frame, dataSetPath)
    return pointCloud

def evaluate_point_cloud(points, k=2, outlier_std_ratio=2.0):
    # Check for invalid values
    mask_valid = np.isfinite(points).all(axis=1)
    valid_points = points[mask_valid]
    n_total = len(points)
    n_valid = len(valid_points)
    n_invalid = n_total - n_valid

    if n_valid < 2:
        raise ValueError("Not enough valid points.")

    # Nearest neighbor distances (k=2 to skip self-distance)
    tree = KDTree(valid_points)
    dists, _ = tree.query(valid_points, k=k)
    nn_dists = dists[:, 1]  # skip distance to self (0.0)

    mean_dist = np.mean(nn_dists)
    std_dist = np.std(nn_dists)

    # Outlier rejection: statistical method
    threshold = 0.0007 + outlier_std_ratio * 0.003 # values determined from high quality datasets
    n_outliers = np.sum(nn_dists > threshold)
    outlier_percentage = 100 * n_outliers / n_valid

    return {
        "n_total": n_total,
        "n_valid": n_valid,
        "n_invalid": n_invalid,
        "mean_nn_dist": mean_dist,
        "std_nn_dist": std_dist,
        "outlier_percentage": outlier_percentage
    }

def print_report_single_frame(frame, dataSetPath, results):
    print(f"Frame: {frame} in dataset: {dataSetPath}")
    print(f" - Total points: {results['n_total']}")
    print(f" - Invalid points: {results['n_invalid']}")
    print(f" - Mean NN distance: {results['mean_nn_dist']:.4f}")
    print(f" - Std NN distance: {results['std_nn_dist']:.4f}")
    print(f" - Outlier percentage: {results['outlier_percentage']:.2f}%\n")

def print_report_dataset(dataSetPath, avg_results):
    print(f"Dataset: {dataSetPath}")
    print(f" - Total points: {int(avg_results['avg_n_total'])}")
    print(f" - Mean NN distance: {avg_results['avg_mean_nn_dist']:.4f}")
    print(f" - Std NN distance: {avg_results['avg_std_nn_dist']:.4f}")
    print(f" - Outlier percentage: {avg_results['avg_outlier_percentage']:.2f}%\n")

# Example usage
if __name__ == "__main__":
    results = []
    for dataSetPath in dataSetPaths:
        num_frames = eval.getNumImageSetsInDataSet(dataSetPath)
        results_per_dataset = []
        n_totals = []
        mean_nn_dists = []
        std_nn_dists = []
        outlier_percentages = []
        for frame in range(0,num_frames):   
            pointCloud = load_point_cloud(frame,dataSetPath)
            points=pointCloud[1]
            points=points[:,:]
            result = evaluate_point_cloud(points)
            n_totals.append(result['n_total'])
            mean_nn_dists.append(result['mean_nn_dist'])
            std_nn_dists.append(result['std_nn_dist'])
            outlier_percentages.append(result['outlier_percentage'])
            if report_single:
                print_report_single_frame(frame, dataSetPath, result)
            if vis:
                ax = plt.figure().add_subplot(projection='3d')
                plotPointCloud(ax, points=pointCloud[0], colors=pointCloud[1])
                plt.show(block=True)
        avg_n_total = np.mean(n_totals)
        avg_mean_nn_dist = np.mean(mean_nn_dists)
        avg_std_nn_dist = np.mean(std_nn_dists)
        avg_outlier_percentage = np.mean(outlier_percentages)
        avg_results = {
        "avg_n_total": avg_n_total,
        "avg_mean_nn_dist": avg_mean_nn_dist,
        "avg_std_nn_dist": avg_std_nn_dist,
        "avg_outlier_percentage": avg_outlier_percentage,
        }
        print_report_dataset(dataSetPath,avg_results)
        results.append(avg_results)
        print()
        