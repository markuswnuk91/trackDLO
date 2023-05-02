#pragma once
#include "trajectory.h"
#include <Eigen/Dense>
#include <string>
#include <vector>

class inverseKinematics
{
public:
    virtual ~inverseKinematics(){};
    Eigen::Matrix<double, 7, 6> pinvPanda(Eigen::Matrix<double, 6, 7> A);
    Eigen::Matrix<double, 7, 6> dampedPinvPanda(Eigen::Matrix<double, 6, 7> A);

    double damping_factor = 1e-3;

    virtual Eigen::Matrix<double, 7, 1>
    calculateIK(Eigen::Matrix<double, 7, 1> jointAngles,
                Eigen::Matrix<double, 6, 1> velocities,
                Eigen::Matrix<double, 6, 7> jacobian) = 0;
};