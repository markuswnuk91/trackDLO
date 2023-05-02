#include "motion.h"
#include "sevenPhaseProfile.h"
#include "trajectory.h"
#include "trajectoryParameters.h"
#include "trajectoryStruct.h"
#include <Eigen/Dense>
#include <iostream>
#include <math.h>
#include <memory>
#include <stdexcept>

std::vector<double> matToVec(Eigen::MatrixXd v1)
{
    std::vector<double> v2;
    v2.resize(v1.size());
    v2 = std::vector<double>(v1.data(), v1.data() + v1.rows() * v1.cols());
    return v2;
};

motion::motion(trajectoryParameters trajParams, sevenPhaseProfile profile,
               bool verbose)
    : _verbose(verbose)
{
    _trajParams = trajParams;
    _profile = std::make_shared<sevenPhaseProfile>(profile);
    _profileRotation = std::make_shared<sevenPhaseProfile>(profile);
};

void motion::setVerbosity(bool verbose) { _verbose = verbose; }

void motion::plot() { std::cout << "Currently not implemented \n"; };

void motion::initializeCallbackJointSpace(Eigen::MatrixXd q_start,
                                          Eigen::MatrixXd q_goal)
{
    std::cout << "Calculate Trajectory from\n"
              << q_start << "\nTo \n"
              << q_goal << "\n";

    _trajectory = trajectory(_profile, q_start, q_goal, _trajParams, _verbose);

    std::function<trajectoryJointSpace(double&)> cbJointSpace =
        [this](double& time) -> trajectoryJointSpace
    {
        this->index_Trajectory = int(time / this->_trajParams.timestep);
        // Can be improved

        if (this->index_Trajectory >
            this->_trajectory.jointPositions.size() - 1)
        {
            this->index_Trajectory =
                this->_trajectory.jointPositions.size() - 1;
            this->trajJointSpace.active = false;
        }

        this->position =
            this->_trajectory.jointPositions.row(this->index_Trajectory)
                .transpose();
        this->vel = this->_trajectory.velocities.row(this->index_Trajectory)
                        .transpose();
        this->acc = this->_trajectory.accelerations.row(this->index_Trajectory)
                        .transpose();

        for (size_t i = 0; i < 7; i++)
        {
            this->trajJointSpace.q[i] = this->position(i);
            this->trajJointSpace.dq[i] = this->vel(i);
            this->trajJointSpace.ddq[i] = this->acc(i);
        }

        return this->trajJointSpace;
    };

    jointcallback = cbJointSpace;
}

void motion::initializeCallback(Eigen::Affine3d startPose,
                                Eigen::Affine3d goalPose)
{
    std::cout << "Calculate Trajectory from\n"
              << startPose.matrix() << "\nTo\n"
              << goalPose.matrix() << "\n";

    _trajectory = trajectory(_profile, _profileRotation, startPose, goalPose,
                             _trajParams, _verbose);

    std::function<trajectoryCartesianSpace(double&)> cbCartSpace =
        [this](double& time) -> trajectoryCartesianSpace
    {
        this->index_Trajectory = int(time / this->_trajParams.timestep);

        if (this->index_Trajectory > this->_trajectory.positions.size() - 1)
        {
            this->index_Trajectory = this->_trajectory.positions.size() - 1;
            this->trajCartSpace.active = false;
        }

        this->pose.matrix() = this->_trajectory.getCartesianPosition(time);
        this->vel = this->_trajectory.getVelocity(time);
        this->acc = this->_trajectory.accelerations.row(this->index_Trajectory);

        // Set Rotation

        this->trajCartSpace.pose[0] = this->pose.rotation()(0, 0);
        this->trajCartSpace.pose[1] = this->pose.rotation()(1, 0);
        this->trajCartSpace.pose[2] = this->pose.rotation()(2, 0);
        this->trajCartSpace.pose[3] = 0.0;

        this->trajCartSpace.pose[4] = this->pose.rotation()(0, 1);
        this->trajCartSpace.pose[5] = this->pose.rotation()(1, 1);
        this->trajCartSpace.pose[6] = this->pose.rotation()(2, 1);
        this->trajCartSpace.pose[7] = 0.0;

        this->trajCartSpace.pose[8] = this->pose.rotation()(0, 2);
        this->trajCartSpace.pose[9] = this->pose.rotation()(1, 2);
        this->trajCartSpace.pose[10] = this->pose.rotation()(2, 2);

        this->trajCartSpace.pose[11] = 0.0;

        // Set X Y Z
        this->trajCartSpace.pose[12] = this->pose.translation()(0);
        this->trajCartSpace.pose[13] = this->pose.translation()(1);
        this->trajCartSpace.pose[14] = this->pose.translation()(2);

        this->trajCartSpace.pose[15] = 1.0;

        for (size_t i = 0; i < 6; i++)
        {
            this->trajCartSpace.v[i] = this->vel(i);
            this->trajCartSpace.a[i] = this->acc(i);
        }

        return this->trajCartSpace;
    };

    cartcallback = cbCartSpace;
}

std::function<trajectoryJointSpace(double&)> motion::getCallbackJointSpace()
{
    return jointcallback;
}

std::function<trajectoryCartesianSpace(double&)>
motion::getCallbackCartesianSpace()
{
    return cartcallback;
}

std::function<trajectoryCartesianSpace(double&)>
motion::callbackTrajectoryCartesianSpace(Eigen::Affine3d startPose,
                                         Eigen::Affine3d goalPose)
{

    std::cout << "Calculate Trajectory from\n"
              << startPose.matrix() << "\nTo\n"
              << goalPose.matrix() << "\n";

    _trajectory = trajectory(_profile, _profileRotation, startPose, goalPose,
                             _trajParams, _verbose);

    std::function<trajectoryCartesianSpace(double&)> cbCartSpace =
        [this](double& time) -> trajectoryCartesianSpace
    {
        this->index_Trajectory = int(time / this->_trajParams.timestep);
        // Can be improved

        if (this->index_Trajectory > this->_trajectory.positions.size() - 1)
        {
            this->index_Trajectory = this->_trajectory.positions.size() - 1;
            this->trajCartSpace.active = false;
        }

        this->pose = this->_trajectory.positions[this->index_Trajectory];
        this->vel = this->_trajectory.velocities.row(this->index_Trajectory)
                        .transpose();
        this->acc = this->_trajectory.accelerations.row(this->index_Trajectory)
                        .transpose();

        // Set Rotation

        this->trajCartSpace.pose[0] = this->pose.rotation()(0, 0);
        this->trajCartSpace.pose[1] = this->pose.rotation()(1, 0);
        this->trajCartSpace.pose[2] = this->pose.rotation()(2, 0);
        this->trajCartSpace.pose[3] = 0.0;

        this->trajCartSpace.pose[4] = this->pose.rotation()(0, 1);
        this->trajCartSpace.pose[5] = this->pose.rotation()(1, 1);
        this->trajCartSpace.pose[6] = this->pose.rotation()(2, 1);
        this->trajCartSpace.pose[7] = 0.0;

        this->trajCartSpace.pose[8] = this->pose.rotation()(0, 2);
        this->trajCartSpace.pose[9] = this->pose.rotation()(1, 2);
        this->trajCartSpace.pose[10] = this->pose.rotation()(2, 2);

        this->trajCartSpace.pose[11] = 0.0;

        // Set X Y Z
        this->trajCartSpace.pose[12] = this->pose.translation()(0);
        this->trajCartSpace.pose[13] = this->pose.translation()(1);
        this->trajCartSpace.pose[14] = this->pose.translation()(2);

        this->trajCartSpace.pose[15] = 1.0;

        for (size_t i = 0; i < 6; i++)
        {
            this->trajCartSpace.v[i] = this->vel(i);
            this->trajCartSpace.a[i] = this->acc(i);
        }

        return this->trajCartSpace;
    };
    return cbCartSpace;
}
trajectory motion::getLinearJointSpaceTrajectory(Eigen::MatrixXd q_start,
                                                 Eigen::MatrixXd q_goal)
{
    _trajectory = trajectory(_profile, q_start, q_goal, _trajParams, _verbose);
    return _trajectory;
}

trajectory motion::getLinearTrajectory(Eigen::Affine3d startPose,
                                       Eigen::Affine3d goalPose)
{

    auto _traj = trajectory(_profile, _profileRotation, startPose, goalPose,
                            _trajParams, _verbose);

    return _traj;
};

trajectory motion::getLinearTrajectory(Eigen::Matrix4d startPose,
                                       Eigen::Matrix4d goalPose)
{
    return getLinearTrajectory(Eigen::Affine3d(startPose),
                               Eigen::Affine3d(goalPose));
}

trajectory::trajectory(bool verbose)
    : _verbose(verbose){

      };
