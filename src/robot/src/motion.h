#pragma once
#include "sevenPhaseProfile.h"
#include "trajectory.h"
#include "trajectoryStruct.h"
#include <Eigen/Dense>
#include <memory>
#include <string>
#include <vector>

class motion
{
public:
    motion(trajectoryParameters trajParams = trajectoryParameters(),
           sevenPhaseProfile profile = sevenPhaseProfile(),
           bool verbose = false);
    void plot();
    void setVerbosity(bool verbose);
    std::function<trajectoryCartesianSpace(double&)>
    callbackTrajectoryCartesianSpace(Eigen::Affine3d startPose,
                                     Eigen::Affine3d goalPose);
    trajectory getLinearTrajectory(Eigen::Affine3d startPose,
                                   Eigen::Affine3d goalPose);
    trajectory getLinearTrajectory(Eigen::Matrix4d startPose,
                                   Eigen::Matrix4d goalPose);

    trajectory getLinearJointSpaceTrajectory(Eigen::MatrixXd q_start,
                                             Eigen::MatrixXd q_goal);

    void initializeCallback(Eigen::Affine3d startPose,
                            Eigen::Affine3d goalPose);
    void initializeCallbackJointSpace(Eigen::MatrixXd q_start,
                                      Eigen::MatrixXd q_goal);
    std::function<trajectoryCartesianSpace(double&)>
    getCallbackCartesianSpace();

    std::function<trajectoryJointSpace(double&)> getCallbackJointSpace();

protected:
    trajectory _trajectory;
    trajectoryParameters _trajParams;
    std::vector<double> matToVec(Eigen::MatrixXd v1);

private:
    trajectoryCartesianSpace trajCartSpace;
    trajectoryJointSpace trajJointSpace;

    int index_Trajectory;
    Eigen::Affine3d pose;
    Eigen::MatrixXd position;
    Eigen::VectorXd vel;
    Eigen::MatrixXd acc;
    std::shared_ptr<motionProfile> _profile;
    std::shared_ptr<motionProfile> _profileRotation;

    std::function<trajectoryCartesianSpace(double&)> cartcallback;
    std::function<trajectoryJointSpace(double&)> jointcallback;
    bool _verbose;
};
