import pickle
import sys
import os

sys.path.append(os.getcwd().replace("/src/evaluation", ""))

filepath_new = "/mnt/d/Dissertation_xwk/Software/trackdlo/data/eval/tracking/results/20230524_171237_ManipulationSequences_mountedWireHarness_modelY/result.pkl"
filepath = "/mnt/d/Dissertation_xwk/Software/trackdlo/data/eval/tracking/results_old/20230524_161235_ManipulationSequences_mountedWireHarness_arena_old/result.pkl"
if __name__ == "__main__":
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    with open(filepath_new, "rb") as f:
        data_new = pickle.load(f)
    print("Done")
