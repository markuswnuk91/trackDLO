#include "dampedInverseKinematics.h"
#include <Eigen/Dense>

dampedInverseKinematics::dampedInverseKinematics()
{

    _Joint_limits << -2.8973, 2.8973, -1.7628, 1.7628, -2.8973, 2.8973, -3.0718,
        -0.0698, -2.8973, 2.8973, -0.0175, 3.7525, -2.8973, 2.8973;
    _q_avg << 0.0, 0.0, 0.0, -1.57, 0.0, 1.8, 0.0;
    _k_p = 1e-3;
}

dampedInverseKinematics::dampedInverseKinematics(
    Eigen::Matrix<double, 7, 2> Joint_limits, Eigen::Matrix<double, 7, 1> q_avg,
    double k_p)
    : _Joint_limits(Joint_limits), _q_avg(q_avg), _k_p(k_p)
{
}

Eigen::Matrix<double, 7, 1>
dampedInverseKinematics::calculateIK(Eigen::Matrix<double, 7, 1> jointAngles,
                                     Eigen::Matrix<double, 6, 1> velocities,
                                     Eigen::Matrix<double, 6, 7> jacobian)
{

    J_pseudoInv = inverseKinematics::dampedPinvPanda(jacobian);
    // J_pseudoInv = inverseKinematics::pinvPanda(jacobian);

    zeroSpacePerformance = this->performanceIndex(jointAngles);

    zeroSp = (Eigen::MatrixXd::Identity(7, 7) - J_pseudoInv * jacobian) *
             zeroSpacePerformance;

    return J_pseudoInv * velocities + zeroSp;
}

Eigen::MatrixXd dampedInverseKinematics::performanceIndex(Eigen::MatrixXd q)
{
    // """
    // Define the Normal Configuration with q_normal
    // """
    return _k_p * (_q_avg - q);
}