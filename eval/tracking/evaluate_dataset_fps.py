import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance_matrix
from warnings import warn
import shutil
from datetime import datetime

try:
    sys.path.append(os.getcwd().replace("/eval", ""))
    from src.evaluation.tracking.trackingEvaluation import TrackingEvaluation
except:
    print("Imports for FPS evaluation script failed.")
    raise

controlOpt = {
    "dataSetsToLoad": [2],
}

dataSetPaths = [
    "data/darus_data_download/data/20230524_171237_ManipulationSequences_mountedWireHarness_modelY/",
    "data/darus_data_download/data/20230524_161235_ManipulationSequences_mountedWireHarness_arena/",
    "data/darus_data_download/data/20230807_162939_ManipulationSequences_mountedWireHarness_partial/",
]


def calculate_time_differences(timestamps):
    # Define the format of the timestamps
    timestamp_format = "%Y%m%d_%H%M%S_%f"

    # Initialize a list to store the time differences in milliseconds
    time_differences = []

    # Loop through the timestamps and calculate the differences between consecutive ones
    for i in range(1, len(timestamps)):
        # Convert the current and previous timestamp strings to datetime objects
        datetime1 = datetime.strptime(timestamps[i - 1], timestamp_format)
        datetime2 = datetime.strptime(timestamps[i], timestamp_format)

        # Calculate the difference between the two datetime objects
        time_diff = datetime2 - datetime1

        # Convert the time difference to milliseconds and append to the list
        time_diff_milliseconds = int(time_diff.total_seconds() * 1000)
        time_differences.append(time_diff_milliseconds)

    return time_differences


def extract_timestamps(filenames):
    timestamps = []

    for filename in filenames:
        # Split the filename to extract the timestamp
        timestamp = filename.split("_image_rgb.png")[0]
        timestamps.append(timestamp)

    return timestamps


if __name__ == "__main__":
    if controlOpt["dataSetsToLoad"][0] == -1:
        dataSetsToEvaluate = dataSetPaths
    else:
        dataSetsToEvaluate = [
            dataSetPath
            for i, dataSetPath in enumerate(dataSetPaths)
            if i in controlOpt["dataSetsToLoad"]
        ]

    for dataSetPath in dataSetsToEvaluate:
        # setup evalulation
        global eval
        eval = TrackingEvaluation()
        frame_names = eval.dataHandler.getDataSetFileNames_RBG(dataSetPath + "data")
        # get the data handler
        time_stamps = extract_timestamps(frame_names)

        time_diffs = calculate_time_differences(time_stamps)

        average_time_diff = np.mean(time_diffs)
        std_time_diff = np.std(time_diffs)
        print(
            "Average time diff between frames of evaluated data sets: {} +- {}".format(
                average_time_diff, std_time_diff
            )
        )
