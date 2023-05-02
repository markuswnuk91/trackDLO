
#pragma once
#include "dampedInverseKinematics.h"
#include "motion.h"
#include "trajectoryStruct.h"
#include "zeroSpaceInverseKinematics.h"
#include <Eigen/Dense>
#include <deque>
#include <franka/control_types.h>
#include <franka/duration.h>
#include <franka/exception.h>
#include <franka/gripper.h>
#include <franka/model.h>
#include <franka/robot.h>
#include <franka/robot_state.h>
#include <mutex>
#include <string>
#include <thread>

class lowLevelControl
{
public:
    enum impedance
    {
        Cartesian,
        JointSpace
    };

    // lowLevelControl();
    lowLevelControl(const std::string& address = "172.16.0.2",
                    inverseKinematics* inverseKin = nullptr);
    ~lowLevelControl();
    bool readOnce(franka::RobotState* roboState);
    void registerRobotStateCallback(
        std::function<void(const franka::RobotState&, double&)>*
            _callbackRoboState);
    void registerTrajectoryCartesianCallback(std::shared_ptr<motion> motion);
    void registerTrajectoryJointCallback(std::shared_ptr<motion> motion);

    Eigen::VectorXd lowPassFilter(Eigen::VectorXd newInput,
                                  Eigen::VectorXd oldInput,
                                  double cutOffFrequency, double timestep);
    Eigen::MatrixXd lowPassFilter(Eigen::MatrixXd newInput,
                                  Eigen::MatrixXd oldInput,

                                  double cutOffFrequency, double timestep);
    void moveCartesian(trajectory _traj);
    void setInverseKinematics(inverseKinematics* inverseKin);
    bool isIdle();

    void emptyQueueAndStopCartesianImpedance();

    void startImpedanceControl(impedance _impedanceType = impedance::Cartesian);
    void stopImpedanceControl();
    void runJointSpaceImpedanceControlLoop();
    // void runCartesianImpedanceControlLoop_Timo_idee();
    void changeImpedance(Eigen::MatrixXd new_k_gains,
                         Eigen::MatrixXd new_d_gains, double new_factorAtan);

    void addJointControl(Eigen::MatrixXd jointVelocityAdd);
    void addCartVelocity(Eigen::MatrixXd cartVelocityAdd);

    franka::Torques _cartControlLoop(const franka::RobotState& robot_state,
                                     franka::Duration period);

    // Gripper functions
    franka::GripperState readGripper();
    void homeGripper();
    void moveGripper(double width, double speed);
    void grasp(double width, double speed, double force,
               double epsilon_inner = 0.005, double epsilon_outer = 0.005);

protected:
    void runCartesianImpedanceControlLoop();
    void setDefaultBehaviour();

    franka::Robot _robot;
    franka::Model pandaModel;

    Eigen::Matrix<double, 7, 2> Joint_limits;
    Eigen::VectorXd q_avg;
    std::function<void(const franka::RobotState&, double&)>*
        callbackRobotState = nullptr;
    std::function<trajectoryJointSpace(double&)> callbackTrajectoryJointspace =
        nullptr;
    std::function<trajectoryCartesianSpace(double&)>
        callbackTrajectoryCartesianSpace = nullptr;

    double controlTime;
    Eigen::VectorXd cart_vels;

    Eigen::MatrixXd cart_pose;
    Eigen::Matrix<double, 7, 6> J_pseudoInv;
    Eigen::Matrix<double, 7, 1> zeroSpacePerformance;
    Eigen::Matrix<double, 7, 1> zeroSp;
    std::array<double, 7> jointvels{};
    bool _stop;
    Eigen::Matrix<double, 7, 7> identity7x7;
    double k_p;
    enum Status
    {
        idle,
        inMotion
    };
    Status robotStatus;
    std::unique_ptr<inverseKinematics> _inverseKinematics;
    dampedInverseKinematics defaultIK;
    // zeroSpaceInverseKinematics defaultIK;

private:
    Eigen::Matrix<double, 6, 1> d_gains;
    Eigen::Matrix<double, 6, 1> k_gains;
    std::string addressGripper;

    Eigen::Matrix<double, 7, 1> d_gains_jointspace;
    Eigen::Matrix<double, 7, 1> k_gains_jointspace;

    Eigen::Matrix<double, 7, 1> _jointVelocityAdd;
    Eigen::Matrix<double, 6, 1> _cartVelocityAdd;

    // Use atan for error fun
    double factor = 100.0;

    void robotCartesianMove();
    std::thread _robotThread;
    std::mutex mutex;
    double time = 0.0;
    int index_Trajectory = 0;
    Eigen::Matrix<double, 6, 7> J_zeroJac;
    Eigen::Matrix<double, 7, 1> nextVelocities;
    bool stop = false;
    double absVelocity = 0.0;
    std::array<double, 7> jointVels{};
    trajectory _trajectory;

    std::deque<std::shared_ptr<motion>> trajectoryQueue;
    std::deque<std::shared_ptr<motion>> trajectoryQueueJointSpace;

    bool joinFromImpedanceControl = false;

    bool startingImpedanceControl = true;
    trajectoryCartesianSpace cartTrajectory;
    std::shared_ptr<motion> currentlyActiveMotion;
};
