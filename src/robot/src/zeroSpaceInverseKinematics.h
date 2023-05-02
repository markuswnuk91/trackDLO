#pragma once
#include "inverseKinematics.h"
#include "trajectory.h"
#include <Eigen/Dense>
#include <string>
#include <vector>

class zeroSpaceInverseKinematics : public inverseKinematics
{
public:
    zeroSpaceInverseKinematics();
    zeroSpaceInverseKinematics(Eigen::Matrix<double, 7, 2> Joint_limits,
                               Eigen::Matrix<double, 7, 1> q_avg, double k_p);
    Eigen::Matrix<double, 7, 1>
    calculateIK(Eigen::Matrix<double, 7, 1> jointAngles,
                Eigen::Matrix<double, 6, 1> velocities,
                Eigen::Matrix<double, 6, 7> jacobian);

private:
    Eigen::Matrix<double, 7, 2> _Joint_limits;
    Eigen::Matrix<double, 7, 1> _q_avg;
    Eigen::Matrix<double, 7, 6> J_pseudoInv;
    Eigen::MatrixXd performanceIndex(Eigen::MatrixXd q);
    Eigen::Matrix<double, 7, 1> zeroSpacePerformance;
    Eigen::Matrix<double, 7, 1> zeroSp;
    Eigen::MatrixXd zeroSpace;
    double _k_p = 1e-3;
};