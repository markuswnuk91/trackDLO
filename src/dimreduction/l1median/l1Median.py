import sys
import os
import numpy as np
from scipy.spatial import distance_matrix
import numbers
from warnings import warn

try:
    sys.path.append(os.getcwd().replace("/src/dimreduction/l1median", ""))
    from src.dimreduction.dimensionalityReduction import DimensionalityReduction
except:
    print("Imports for L1-Median failed.")
    raise


class L1Median(DimensionalityReduction):
    """
    Implementation according to the Paper
    "Huang et al.: L1-Medial Skeleton of Point Cloud, ACM Transactions on Graphics, 32(4):1, 2013"
    Attributes:
    -------------
    Y: Jx3 np.ndarray
        pointCloud the skeleton line should be extracted from
    T: Ix3 np.ndarray
        seedpoints used to represent the sought centerline
    h: float
        support radius h defining the size of the supporting local neighborhood for L1-medial skeleton
    mu: float
        weighting parameter for repulsion force between seedpoints
    """

    def __init__(self, h=None, mu=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.h = 0.12 if h is None else h
        self.mu = 0.35 if mu is None else mu
        self.iteration = 0

    def get_h0(self, points):
        x_max = points[:, 0].max()
        x_min = points[:, 0].min()

        y_max = points[:, 1].max()
        y_min = points[:, 1].min()

        z_max = points[:, 2].max()
        z_min = points[:, 2].min()

        diagonal = (
            (x_max - x_min) ** 2 + (y_max - y_min) ** 2 + (z_max - z_min) ** 2
        ) ** 0.5

        Npoints = len(points)

        return 2 * diagonal / (Npoints ** (1.0 / 3))

    def get_thetas(self, r, h):
        """
        INPUT:
            r: NxM np.ndarray of distances given by a distance matrix of two datasets NxK and MxK, where N and M are number of points and K are dimensions
            h: support radius h defining the size of the supporting local neighborhood for L1-medial skeleton

        OUTPUT:
            thetas: NXM np.ndarray of weighting factors for each point correspondence
        """
        thetas = np.exp((-(r**2)) / ((h / 2) ** 2))
        # Clip to JUST not zero
        # thetas =  np.clip(thetas, 10**-323, None)
        return thetas

    def get_alphas(self, X, Y, h):
        """
        INPUT:
            Y:      Jx3 np.ndarray, pointCloud
            X:      Ix3 np.ndarray, seedpoints to represent the sought centerline
            h: support radius h defining the size of the supporting local neighborhood for L1-medial skeleton
        """
        distances = distance_matrix(X, Y)
        thetas = self.get_thetas(distances, h)
        alphas = np.divide(
            thetas,
            distances,
            out=np.zeros(thetas.shape, dtype=float),
            where=distances != 0,
        )
        return alphas

    def get_betas(self, X, h):
        """
        INPUT:
            X:      Ix3 np.ndarray, seedpoints to represent the sought centerline
            h: support radius h defining the size of the supporting local neighborhood for L1-medial skeleton
        """
        distances = distance_matrix(X, X)
        thetas = self.get_thetas(distances, h)
        distances2 = distances * distances
        betas = np.divide(
            thetas,
            distances2,
            out=np.zeros(thetas.shape, dtype=float),
            where=distances2 != 0,
        )
        return betas

    def get_sigmas(self, X, h):
        """
        INPUT:
            X:      Ix3 np.ndarray, seedpoints to represent the sought centerline
            h: support radius h defining the size of the supporting local neighborhood for L1-medial skeleton

        OUTPUT:
            sigmas: Ix1 vector of sigmas
        """
        thetas = self.get_thetas(distance_matrix(X, X), h)
        sigmas = np.zeros(len(X))
        for i, x in enumerate(X):
            C_i = np.zeros((3, 3))
            for i_dash, x_dash in enumerate(np.delete(X, i, 0)):
                C_i += thetas[i, i_dash] * np.outer(x, x_dash)
            lambdas, eigVecs = np.linalg.eig(C_i)

            if np.iscomplex(lambdas).any():
                # print("Found complex eigenvalues.")
                lambdas = np.real(lambdas)
            sigmas[i] = np.amax(lambdas) / np.sum(lambdas)
        return sigmas

    def calculateReducedRepresentation(self):
        """
        Function to perform L1 Median estimation.
        """
        # ---------------------------------#
        # begin algorithm
        # ---------------------------------#
        J = len(self.Y)  # number of input points
        I = len(self.T)  # number of seedpoints

        alpha_matrix = np.zeros((I, J))
        beta_matrix = np.zeros((I, I))

        while self.iteration < self.max_iterations:
            alpha_matrix = self.get_alphas(self.T, self.Y, self.h)
            beta_matrix = self.get_betas(self.T, self.h)
            sigmas = self.get_sigmas(self.T, self.h)
            sum_J_qj_aij = np.column_stack(
                (
                    # sum over x dimension
                    np.sum(alpha_matrix * self.Y[:, 0].transpose(), axis=1),
                    # sum over y dimension
                    np.sum(alpha_matrix * self.Y[:, 1].transpose(), axis=1),
                    # sum over z dimension
                    np.sum(alpha_matrix * self.Y[:, 2].transpose(), axis=1),
                )
            )
            sum_J_aij = np.sum(alpha_matrix, axis=1)
            term1 = sum_J_qj_aij / sum_J_aij[:, None]  # mean shift term

            sum_Id_xixid_betaid = np.sum(
                (
                    # repeat points along first dimenstion to make cube of I x I x 3 and subtract transposed Ix3xI cube
                    np.tile(self.T, (I, 1, 1))
                    - np.transpose(np.tile(self.T, (I, 1, 1)), (1, 0, 2))
                )
                # multiply with beta factor along first dimension
                * np.transpose(np.tile(beta_matrix, (3, 1, 1)), (2, 1, 0))
                # sum over first dimension to obtain agian Ix3 array
                ,
                axis=0,
            )
            sum_Id_betaiid = np.sum(beta_matrix, axis=1) - np.diag(beta_matrix)
            term2 = (
                self.mu
                * sigmas[:, None]
                * (sum_Id_xixid_betaid / sum_Id_betaiid[:, None])
            )  # regularization term

            # update positions
            self.T = term1 + term2
            self.iteration += 1

            if callable(self.callback):
                self.callback()

        return self.T
