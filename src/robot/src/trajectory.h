#pragma once
#include "motionProfile.h"
#include "trajectoryParameters.h"
#include <Eigen/Dense>
#include <math.h>
#include <memory>
#include <string>
#include <vector>

class trajectory
{
public:
    trajectory(bool verbose = true);
    trajectory(std::shared_ptr<motionProfile> mProfileTranslation,

               std::shared_ptr<motionProfile> mProfileRotation,
               Eigen::Affine3d startPose, Eigen::Affine3d goalPose,
               trajectoryParameters motionParams, bool verbose = false);
    trajectory(std::shared_ptr<motionProfile> mProfile, Eigen::MatrixXd q_start,
               Eigen::MatrixXd q_goal, trajectoryParameters motionParams,
               bool verbose = false);
    Eigen::VectorXd getVelocity(double& time);
    Eigen::MatrixXd getCartesianPosition(double& time);
    Eigen::VectorXd getJointPosition(double& time);

    Eigen::MatrixXd getAccelerations();
    Eigen::MatrixXd getVelocities();
    std::vector<Eigen::Matrix4d> getPoses();

    Eigen::MatrixXd accelerations;
    Eigen::MatrixXd velocities;
    Eigen::MatrixXd jointPositions;

    std::vector<Eigen::Affine3d> positions;

    void printTrajectoryParameters(trajectoryParameters params);
    bool isInit();

protected:
    bool isInitialized = false;
    bool _verbose;
    bool cartesian;
    Eigen::VectorXd direction_jointspace;
    Eigen::VectorXd joint_position_initial;
    Eigen::Vector3d direction_cartesian_translation;
    Eigen::Vector3d direction_cartesian_rotation;
    std::shared_ptr<motionProfile> _motionProfile;
    std::shared_ptr<motionProfile> _motionProfileRotation;
    Eigen::Affine3d _startPose;
    Eigen::Affine3d _goalPose;
    double length_r;
};