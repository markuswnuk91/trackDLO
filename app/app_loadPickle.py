import pickle
import sys
import os

sys.path.append(os.getcwd().replace("/src/evaluation", ""))

filepath = "/mnt/c/Users/marku/Documents/Dissertation/Software/trackdlo/data/eval/graspingAccuracy/results/20230522_130903_modelY/result.pkl"
if __name__ == "__main__":
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    print("Done")
