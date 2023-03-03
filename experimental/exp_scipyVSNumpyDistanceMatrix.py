# script compares the runtime of manually implemented distance matrix vs. scipy implementation
import os
import sys
import time
from scipy.spatial import distance_matrix
import numpy as np

try:
    sys.path.append(os.getcwd().replace("/experimental", ""))
    from src.utils.utils import sqdistance_matrix
except:
    print("Imports for CPD failed.")
    raise

if __name__ == "__main__":
    X = np.random.rand(1000, 3)
    Y = np.random.rand(10000, 3)

    st_scipy = time.time()
    scipy = np.square(distance_matrix(X, Y))
    et_scipy = time.time()
    t_scipy = et_scipy - st_scipy
    print("Time scipy {}".format(t_scipy))

    st_np = time.time()
    numpy = sqdistance_matrix(X, Y)
    et_np = time.time()

    t_np = et_np - st_np
    print("Time numpy {}".format(t_np))

    print("Cumulated error between methods: {}".format(np.sum(scipy - numpy)))
