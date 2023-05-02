
#pragma once

#include "lowLevelControl.h"
#include "motion.h"
#include <Eigen/Dense>
#include <chrono>
#include <franka/control_types.h>
#include <franka/duration.h>
#include <franka/exception.h>
#include <franka/model.h>
#include <franka/robot.h>
#include <franka/robot_state.h>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <string>

class highLevelControl
{
public:
    highLevelControl(const std::string& address = "172.16.0.2");
    void print_O_T_EE_and_q();
    std::array<double, 16> get_O_T_EE();
    void startTest(Eigen::Matrix4d goalPose4x4);
    franka::RobotState getRobotState();
    void moveToPositionVelocity(Eigen::Matrix4d goalPose4x4,
                                trajectoryParameters params);
    void moveToPosition(Eigen::Matrix4d goalPose4x4,
                        trajectoryParameters params = trajectoryParameters(),
                        bool emptyQueue = false);
    void moveJointSpace(Eigen::MatrixXd q_goal,
                        trajectoryParameters params = trajectoryParameters());

    void startImpedanceControl();
    void startJointImpedanceControl();
    void stopImpedanceControl();
    bool isIdle();

    void setImpedance(Eigen::MatrixXd new_k_gains, Eigen::MatrixXd new_d_gains,
                      double new_factorAtan);

    void addJointControl(Eigen::MatrixXd jointVelocityAdd);
    void addCartVelocity(Eigen::MatrixXd cartVelocityAdd);

    // All Functions to call
    void moveCartesian(Eigen::Matrix4d goalPose4x4);

    // Gripper functions
    franka::GripperState readGripper();
    void homeGripper();
    void moveGripper(double width, double speed);
    void grasp(double width, double speed, double force,
               double epsilon_inner = 0.005, double epsilon_outer = 0.005);

private:
    bool autorun = true;
    bool hasTrajectory = false;
    franka::RobotState _robotState;
    lowLevelControl lowLvlCtl;
    std::function<void(const franka::RobotState&, double&)> stateCallback;
    trajectory _trajectory;
    bool check4x4Matrix(Eigen::Matrix4d mat);
    motion _motion;
    bool cartesianImpedanceIsRunning = false;
    bool jointImpedanceIsRunning = false;
    bool initializedLastPosition = false;
    bool initLastJointPosition = false;
    Eigen::Affine3d goalPose;
    Eigen::Affine3d startPose;
    Eigen::MatrixXd q_start, q_temp;
};