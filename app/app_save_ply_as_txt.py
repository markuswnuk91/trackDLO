import os, sys
import numpy as np

try:
    sys.path.append(os.getcwd().replace("/app", ""))
    from src.sensing.loadPointCloud import readPointCloudFromPLY
except:
    print("Imports for application topologyReconstructionFromPointCloud failed.")
    raise

# control parameters
save = True  # if data  should be saved under the given savepath
savePath = "tests/testdata/topologyExtraction/wireHarness.txt"

dataPath = "data/darus_data_download/data/dlo_dataset/DLO_Data/20220203_Random_Poses_Unfolded_Wire_Harness/pointcloud_2.ply"

downsamplingInputRatio = 1 / 3

if __name__ == "__main__":
    if save:
        pointSet = readPointCloudFromPLY(dataPath)[
            :: int((1 / downsamplingInputRatio)), :3
        ]
        np.savetxt(savePath, pointSet)
    else:
        print("Set 'save' to True, to save the data set")
