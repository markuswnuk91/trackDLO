#script compares the runtime of manually implemented gaussian_kernel difference matrix vs. scipy inpmentation
import time

from scipy.spatial import distance_matrix
import numpy as np

def gaussian_kernel(X, beta, Y=None):
    """
    Calculate gaussian kernel matrix.
    Attributes
    ----------
    X: numpy array
        NxD array of points for creating gaussian.
    
    beta: float
        Width of the Gaussian kernel.
    
    Y: numpy array, optional
        MxD array of secondary points to calculate
        kernel with. Used if predicting on points
        not used to train.
        
    Returns
    -------
    K: numpy array
        Gaussian kernel matrix.
            NxN if Y is None
            NxM if Y is not None
    """
    if Y is None:
        Y = X
    diff = X[:, None, :] - Y[None, :,  :]
    diff = np.square(diff)
    diff = np.sum(diff, 2)
    return np.exp(-diff / (2 * beta**2))

if __name__=="__main__":
    X= np.random.rand(1000,3)
    Y= np.random.rand(10000,3)

    st_scipy = time.time()
    scipy = np.exp(-np.square(distance_matrix(X,Y))/ (2 * 1**2))
    et_scipy = time.time()

    t_scipy = et_scipy - st_scipy
    print("Time scipy {}".format(t_scipy))
    st_np = time.time()
    numpy = gaussian_kernel(X,1,Y)
    et_np = time.time()

    t_np = et_np- st_np
    print("Time numpy {}".format(t_np))
    
    print("Cumulated error between methods: {}".format(np.sum(scipy-numpy)) )


