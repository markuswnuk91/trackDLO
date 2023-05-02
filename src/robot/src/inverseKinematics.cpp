#include "inverseKinematics.h"
#include <Eigen/Dense>
#include <iostream>

Eigen::Matrix<double, 7, 6>
inverseKinematics::pinvPanda(Eigen::Matrix<double, 6, 7> A)
{
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU |
                                                 Eigen::ComputeFullV); // M=USV*
    double pinvtoler = 1.e-8; // tolerance
    int row = A.rows();
    int col = A.cols();
    int k = std::min(row, col);
    Eigen::MatrixXd X = Eigen::MatrixXd::Zero(col, row);
    Eigen::MatrixXd singularValues_inv = svd.singularValues(); // singular value
    Eigen::MatrixXd singularValues_inv_mat = Eigen::MatrixXd::Zero(col, row);
    for (long i = 0; i < k; ++i)
    {
        if (singularValues_inv(i) > pinvtoler)
            singularValues_inv(i) = 1.0 / singularValues_inv(i);
        else
            singularValues_inv(i) = 0;
    }
    for (long i = 0; i < k; ++i)
    {
        singularValues_inv_mat(i, i) = singularValues_inv(i);
    }
    X = (svd.matrixV()) * (singularValues_inv_mat) *
        (svd.matrixU().transpose()); // X=VS+U*

    return X;
};

Eigen::Matrix<double, 7, 6>
inverseKinematics::dampedPinvPanda(Eigen::Matrix<double, 6, 7> A)
{
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU |
                                                 Eigen::ComputeFullV); // M=USV*
    double pinvtoler = 1.e-8; // tolerance
    int row = A.rows();
    int col = A.cols();
    int k = std::min(row, col);
    Eigen::MatrixXd X = Eigen::MatrixXd::Zero(col, row);
    Eigen::MatrixXd singularValues_inv = svd.singularValues(); // singular value
    Eigen::MatrixXd singularValues_inv_mat = Eigen::MatrixXd::Zero(col, row);
    for (long i = 0; i < k; ++i)
    {
        if (singularValues_inv(i) > pinvtoler)
            singularValues_inv(i) = singularValues_inv(i);
        else
            singularValues_inv(i) = 0;
    }
    for (long i = 0; i < k; ++i)
    {
        singularValues_inv_mat(i, i) =
            singularValues_inv(i) /
            (singularValues_inv(i) * singularValues_inv(i) + damping_factor);
    }
    X = (svd.matrixV()) * (singularValues_inv_mat) *
        (svd.matrixU().transpose()); // X=VSU*

    // std::cout << "Damped PseudoInv: \n" << X << "\n";

    return X;
};