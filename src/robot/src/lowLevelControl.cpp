#include "lowLevelControl.h"
#include "inverseKinematics.h"
#include <Eigen/Dense>
#include <chrono>
#include <franka/control_types.h>
#include <franka/duration.h>
#include <franka/exception.h>
#include <franka/model.h>
#include <franka/rate_limiting.h>
#include <franka/robot.h>
#include <franka/robot_state.h>
#include <iostream>
#include <math.h>
#include <memory>
#include <mutex>
#include <stdlib.h>
#include <thread>

lowLevelControl::lowLevelControl(const std::string& address,
                                 inverseKinematics* inverseKin)
    : _robot(address), pandaModel(franka::Model(_robot.loadModel())), q_avg(7),
      _inverseKinematics(std::unique_ptr<inverseKinematics>()),
      addressGripper(address)
{
    robotStatus = Status::idle;

    Joint_limits << -2.8973, 2.8973, -1.7628, 1.7628, -2.8973, 2.8973, -3.0718,
        -0.0698, -2.8973, 2.8973, -0.0175, 3.7525, -2.8973, 2.8973;

    q_avg << 0.0, 0.0, 0.0, -1.57, 0.0, 1.8, 0.0;

    _jointVelocityAdd << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    _cartVelocityAdd << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;

    // Set gains for the cartesian impedance control.
    // Stiffness
    k_gains << 250.0, 250.0, 250.0, 20.0, 20.0, 20.0;

    // Damping
    d_gains << 5.0, 5.0, 5.0, 12.0, 12.0, 12.0;

    if (inverseKin != nullptr)
    {
        setInverseKinematics(inverseKin);
    }
    else
    {
        // Set Standard zerospace inverse kinematcis
        // defaultIK = zeroSpaceInverseKinematics(Joint_limits, q_avg, 0.5);
        defaultIK = dampedInverseKinematics(Joint_limits, q_avg, 0.5);
        setInverseKinematics(&defaultIK);
    }

    std::cout << "Initialized Robot "
              << "\n";
    identity7x7 = Eigen::MatrixXd::Identity(7, 7);
};

franka::GripperState lowLevelControl::readGripper()
{
    auto _gripper = franka::Gripper(addressGripper);
    return _gripper.readOnce();
}

void lowLevelControl::homeGripper()
{
    auto _gripper = franka::Gripper(addressGripper);
    _gripper.homing();
}

void lowLevelControl::moveGripper(double width, double speed)
{
    auto _gripper = franka::Gripper(addressGripper);
    _gripper.move(width, speed);
}

void lowLevelControl::grasp(double width, double speed, double force,
                            double epsilon_inner, double epsilon_outer)
{
    auto _gripper = franka::Gripper(addressGripper);
    _gripper.grasp(width, speed, force, epsilon_inner, epsilon_outer);
}

bool lowLevelControl::isIdle()
{
    if (robotStatus == Status::idle)
    {
        return true;
    }
    return false;
}

bool lowLevelControl::readOnce(franka::RobotState* roboState)
{
    if (robotStatus == Status::idle)
    {
        *roboState = _robot.readOnce();
        return true;
    }
    std::cout << "Status not idle! Can't Read Robot State!\n";
    return false;
};

void lowLevelControl::setInverseKinematics(inverseKinematics* inverseKin)
{
    this->_inverseKinematics = std::unique_ptr<inverseKinematics>(inverseKin);
}

void lowLevelControl::setDefaultBehaviour()
{
    // Set the collision behavior.
    std::array<double, 7> lower_torque_thresholds_nominal{
        {25.0, 25.0, 22.0, 20.0, 19.0, 17.0, 14.}};
    std::array<double, 7> upper_torque_thresholds_nominal{
        {35.0, 35.0, 32.0, 30.0, 29.0, 27.0, 24.0}};
    std::array<double, 7> lower_torque_thresholds_acceleration{
        {25.0, 25.0, 22.0, 20.0, 19.0, 17.0, 14.0}};
    std::array<double, 7> upper_torque_thresholds_acceleration{
        {35.0, 35.0, 32.0, 30.0, 29.0, 27.0, 24.0}};
    std::array<double, 6> lower_force_thresholds_nominal{
        {30.0, 30.0, 30.0, 25.0, 25.0, 25.0}};
    std::array<double, 6> upper_force_thresholds_nominal{
        {40.0, 40.0, 40.0, 35.0, 35.0, 35.0}};
    std::array<double, 6> lower_force_thresholds_acceleration{
        {30.0, 30.0, 30.0, 25.0, 25.0, 25.0}};
    std::array<double, 6> upper_force_thresholds_acceleration{
        {40.0, 40.0, 40.0, 35.0, 35.0, 35.0}};

    _robot.setCollisionBehavior(
        lower_torque_thresholds_acceleration,
        upper_torque_thresholds_acceleration, lower_torque_thresholds_nominal,
        upper_torque_thresholds_nominal, lower_force_thresholds_acceleration,
        upper_force_thresholds_acceleration, lower_force_thresholds_nominal,
        upper_force_thresholds_nominal);

    _robot.automaticErrorRecovery();
    _robot.setJointImpedance({{3000, 3000, 3000, 2500, 2500, 2000, 2000}});
}

void lowLevelControl::moveCartesian(trajectory _traj)
{
    if (_robotThread.joinable())
    {
        std::cout << "Joining Robot Thread again\n";
        _robotThread.join();
        robotStatus = Status::idle;
        std::cout << "Successfully joined robot Thread\n";
    }

    switch (robotStatus)
    {
        case idle:
            _trajectory = _traj;
            time = 0.0;

            if (_inverseKinematics)
            {
                setDefaultBehaviour();
                robotStatus = Status::inMotion;
                // Move
                _robotThread =
                    std::thread(&lowLevelControl::robotCartesianMove, this);
            }

            else
            {
                std::cout << "No inverse Kinematics defined. Can't move.\n";
            }

            break;
        case inMotion:
            std::cout << "Abord. Wait for Motion to be finished.\n";
            break;
    }
}

void lowLevelControl::startImpedanceControl(impedance _impedanceType)
{
    if (_robotThread.joinable())
    {
        std::cout << "Joining Robot Thread again\n";
        _robotThread.join();
        robotStatus = Status::idle;
        std::cout << "Successfully joined robot Thread\n";
    }

    switch (robotStatus)
    {
        case idle:

            if (_impedanceType == impedance::Cartesian)
            {

                if (_inverseKinematics)
                {
                    setDefaultBehaviour();
                    robotStatus = Status::inMotion;

                    // Start Controller
                    _robotThread = std::thread(
                        &lowLevelControl::runCartesianImpedanceControlLoop,
                        this);
                }

                else
                {
                    std::cout << "No inverse Kinematics defined. Can't move.\n";
                }
            }

            else if (_impedanceType == impedance::JointSpace)
            {
                setDefaultBehaviour();
                robotStatus = Status::inMotion;

                // Start Controller
                _robotThread = std::thread(
                    &lowLevelControl::runJointSpaceImpedanceControlLoop, this);
            }
            else
            {
                std::cout << "Error setting Impedance Control Loop.\n";
            }

            break;
        case inMotion:
            std::cout << "Abord. Wait for Motion to be finished.\n";
            break;
    }
}

void lowLevelControl::stopImpedanceControl()
{
    if (robotStatus == Status::inMotion)
    {
        joinFromImpedanceControl = true;
        std::cout << "Joining Robot Thread again\n";

        _robotThread.join();
        std::cout << "Successfully joined robot Thread\n";
        robotStatus = Status::idle;
        joinFromImpedanceControl = false;
    }
    else
    {
        std::cout << "No Motion registered. Cant stop Impedance.\n";
    }
}

void lowLevelControl::runJointSpaceImpedanceControlLoop()
{

    trajectoryJointSpace trajectory;
    double time;
    time = 0.0;
    trajectory.active = false;
    bool firstStart = true;

    // Set gains for the join space impedance control.
    // Stiffness
    k_gains_jointspace << 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0;

    // k_gains << 400.0, 400.0, 400.0, 20.0, 20.0, 20.0;

    // Damping
    d_gains_jointspace << 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0;

    // d_gains << 250.0, 250.0, 250.0, 12.0, 12.0, 12.0;

    // Define callback for the joint torque control loop.
    std::function<franka::Torques(const franka::RobotState&, franka::Duration)>
        impedance_control_callback =
            [this, &time, &firstStart,
             &trajectory](const franka::RobotState& robot_state,
                          franka::Duration period) -> franka::Torques
    {
        //  RT Control Loop
        this->mutex.lock();

        if (firstStart ||
            !_jointVelocityAdd.isApprox(Eigen::MatrixXd::Zero(7, 1)))
        {
            firstStart = false;
            trajectory.q = robot_state.q;
        }
        else
        {
            for (size_t i = 0; i < 7; i++)
            {
                trajectory.q[i] += _jointVelocityAdd[i];
            }
        }

        if (this->callbackRobotState)
        {
            (*this->callbackRobotState)(robot_state, this->time);
        }

        if (this->callbackTrajectoryJointspace != nullptr)
        {
            // Update Trajectory
            trajectory = (this->callbackTrajectoryJointspace)(this->time);
            if (trajectory.active == false)
            {
                std::cout << "\nTrajectory inactive\n";
                this->callbackTrajectoryCartesianSpace = nullptr;
            }
        }
        else
        {
            this->time = 0.0;
            trajectory.dq = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
            trajectory.ddq = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        }

        // Set new Callback
        if (this->trajectoryQueueJointSpace.size() > 0 &&
            trajectory.active == false)
        {
            // Try Setting new Callback
            std::cout << "Setting new Trajectory\n";
            this->callbackTrajectoryJointspace =
                this->trajectoryQueueJointSpace.front()
                    ->getCallbackJointSpace();

            this->trajectoryQueueJointSpace.pop_front();
            std::cout << "Popping Front \n";
        }

        // Update time.
        this->time += period.toSec();

        Eigen::Matrix<double, 7, 1> error;

        Eigen::Map<const Eigen::Matrix<double, 7, 1>> q(robot_state.q.data());
        Eigen::Map<const Eigen::Matrix<double, 7, 1>> q_d(trajectory.q.data());

        error = q_d - q;

        for (size_t i = 0; i < 7; i++)
        {
            error[i] = atan(this->factor * error[i]) / this->factor;
        }

        // Use a Cartesian PD-Control with tao (joints) = jacobian.T @ F
        // (cart)

        Eigen::Map<const Eigen::Matrix<double, 7, 1>> dq(robot_state.dq.data());
        Eigen::Map<const Eigen::Matrix<double, 7, 1>> dq_d(
            trajectory.dq.data());

        // Add Custom Velocity override
        Eigen::Matrix<double, 7, 1> tao_PD =
            this->k_gains_jointspace.cwiseProduct(error) +
            this->d_gains_jointspace.cwiseProduct((dq_d)-dq);

        // Read current coriolis terms from model.
        Eigen::Map<const Eigen::Matrix<double, 7, 1>> coriolis(
            this->pandaModel.coriolis(robot_state).data());

        Eigen::Matrix<double, 7, 1> tau_d_calculated = coriolis + tao_PD;

        std::array<double, 7> tau_d_array{};
        Eigen::VectorXd::Map(&tau_d_array[0], 7) = tau_d_calculated;
        this->mutex.unlock();

        if (this->joinFromImpedanceControl)
        {
            return franka::MotionFinished(franka::Torques(tau_d_array));
        }

        return tau_d_array;
    };
    try
    {
        // Start real-time control loop.
        _robot.control(impedance_control_callback);
    }
    catch (const franka::Exception& ex)
    {
        std::cout << "Robot Control Loop failed.\n";
        std::cerr << ex.what() << std::endl;
        this->robotStatus = Status::idle;
    }
}

void lowLevelControl::emptyQueueAndStopCartesianImpedance()
{

    std::this_thread::sleep_for(std::chrono::milliseconds(5));

    bool tryagain = true;
    while (tryagain)
    {
        {
            if (mutex.try_lock())
            {

                trajectoryQueue.clear();
                this->callbackTrajectoryCartesianSpace = nullptr;
                this->cartTrajectory.active = false;
                this->startingImpedanceControl = true;

                mutex.unlock();
                break;
            }
        }
    }
}

franka::Torques
lowLevelControl::_cartControlLoop(const franka::RobotState& robot_state,
                                  franka::Duration period)
{
    //  RT Control Loop
    this->mutex.lock();

    if (this->startingImpedanceControl)
    {
        this->startingImpedanceControl = false;
        this->cartTrajectory.pose = robot_state.O_T_EE;
    }

    if (this->callbackRobotState)
    {
        (*this->callbackRobotState)(robot_state, this->time);
    }

    if (this->callbackTrajectoryCartesianSpace != nullptr)
    {
        // Update Trajectory
        this->cartTrajectory =
            (this->callbackTrajectoryCartesianSpace)(this->time);
        if (this->cartTrajectory.active == false)
        {
            std::cout << "Trajectory inactive\n";
            this->callbackTrajectoryCartesianSpace = nullptr;
        }
    }
    else
    {
        this->time = 0.0;
        this->cartTrajectory.v = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        this->cartTrajectory.a = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    }

    Eigen::Map<const Eigen::Matrix<double, 7, 1>> dq(robot_state.dq.data());

    // Set new Callback
    if (this->trajectoryQueue.size() > 0 &&
        this->cartTrajectory.active == false && dq.norm() < 5.0 / 1000.0)
    {
        // Try Setting new Callback
        std::cout << "Setting new Trajectory\n";
        this->currentlyActiveMotion = this->trajectoryQueue.front();
        this->callbackTrajectoryCartesianSpace =
            this->currentlyActiveMotion->getCallbackCartesianSpace();

        this->trajectoryQueue.pop_front();
        std::cout << "Popping Front \n";
    }

    // Update time.
    this->time += period.toSec();

    // Calculate Torques
    this->J_zeroJac = Eigen::Map<const Eigen::Matrix<double, 6, 7>>(
        this->pandaModel.zeroJacobian(franka::Frame::kEndEffector, robot_state)
            .data());

    // compute error to desired equilibrium pose
    // position error
    Eigen::Affine3d affine_position_d(
        Eigen::Matrix4d::Map(cartTrajectory.pose.data()));
    Eigen::Vector3d position_d(affine_position_d.translation());

    Eigen::Affine3d affine_position(
        Eigen::Matrix4d::Map(robot_state.O_T_EE.data()));
    Eigen::Vector3d position(affine_position.translation());

    Eigen::Matrix<double, 6, 1> error;
    error.head(3) << position_d - position;

    if (error.head(3).norm() > 5.0 / 1000.0)
    {
        error.head(3) << (position_d - position) / error.head(3).norm() * 5.0 /
                             1000.0;
    }

    // orientation error
    // "difference" quaternion
    Eigen::Quaterniond orientation_d(affine_position_d.linear());
    Eigen::Quaterniond orientation(affine_position.linear());

    if (orientation_d.coeffs().dot(orientation.coeffs()) < 0.0)
    {
        orientation.coeffs() << -orientation.coeffs();
    }

    // "difference" quaternion
    Eigen::Quaterniond error_quaternion(orientation.inverse() * orientation_d);

    error.tail(3) << error_quaternion.x(), error_quaternion.y(),
        error_quaternion.z();

    if (error.tail(3).norm() > 250.0 / 1000.0)
    {
        error.tail(3) << error.tail(3) / error.tail(3).norm() * 20.0 / 1000.0;
    }

    // Transform to base frame
    error.tail(3) << affine_position.linear() * error.tail(3);

    for (size_t i = 0; i < 6; i++)
    {
        if (error[i] != 0.0)
        {
            error[i] = atan(this->factor * error[i] / abs(error[i]) *
                            sqrt(abs(error[i]))) /
                       this->factor;
        }
        else
        {
            error[i] = 0.0;
        }
    }

    // Use a Cartesian PD-Control with tao (joints) = jacobian.T @ F
    // (cart)

    Eigen::Map<const Eigen::Matrix<double, 6, 1>> v_d(cartTrajectory.v.data());

    // Add Custom Velocity override
    Eigen::Matrix<double, 7, 1> tao_PD =
        this->J_zeroJac.transpose() *
        (this->k_gains.cwiseProduct(error) +
         this->d_gains.cwiseProduct((v_d + this->_cartVelocityAdd) -
                                    this->J_zeroJac * dq));

    // std::cout << "For err\nv_d:" << v_d << "\nerr" << error << "\nk"
    //           << this->k_gains << "d" << this->d_gains << "\n";

    // Read current coriolis terms from model.
    Eigen::Map<const Eigen::Matrix<double, 7, 1>> coriolis(
        this->pandaModel.coriolis(robot_state).data());

    // Read current mass matrix terms from model.
    Eigen::Map<const Eigen::Matrix<double, 7, 7>> mass(
        this->pandaModel.mass(robot_state).data());

    // Read current gravity terms from model.
    Eigen::Map<const Eigen::Matrix<double, 7, 1>> gravity(
        this->pandaModel.gravity(robot_state).data());

    // Read current desired Accelerations
    Eigen::Map<const Eigen::Matrix<double, 7, 1>> ddq_d(
        robot_state.ddq_d.data());

    auto tao_IK_zero = this->_inverseKinematics->calculateIK(
        Eigen::Map<const Eigen::Matrix<double, 7, 1>>(robot_state.q.data()),
        (Eigen::MatrixXd(6, 1) << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0).finished(),
        this->J_zeroJac);

    // Control Law: tao_q_d + M(q) * ddq_d +
    // C(q,dq) + G(q)
    double factor_zerospace = 6.0;
    Eigen::Matrix<double, 7, 1> tau_d_calculated =
        coriolis + tao_PD + factor_zerospace * tao_IK_zero; // + mass * ddq_d;
    /*+gravity */
    //;

    std::array<double, 7> tau_d_array{};
    Eigen::VectorXd::Map(&tau_d_array[0], 7) = tau_d_calculated;
    this->mutex.unlock();

    if (this->joinFromImpedanceControl)
    {
        return franka::MotionFinished(franka::Torques(tau_d_array));
    }

    return tau_d_array;
}

void lowLevelControl::runCartesianImpedanceControlLoop()
{

    time = 0.0;
    cartTrajectory.active = false;

    this->startingImpedanceControl = true;

    // Define callback for the joint torque control loop.
    std::function<franka::Torques(const franka::RobotState&, franka::Duration)>
        impedance_control_callback =
            [this](const franka::RobotState& robot_state,
                   franka::Duration period) -> franka::Torques
    { return _cartControlLoop(robot_state, period); };

    try
    {
        // Start real-time control loop.
        _robot.control(impedance_control_callback);
    }
    catch (const franka::Exception& ex)
    {
        std::cout << "Robot Control Loop failed.\n";
        std::cerr << ex.what() << std::endl;
        this->robotStatus = Status::idle;
    }
}

lowLevelControl::~lowLevelControl()
{
    if (_robotThread.joinable())
    {
        _robotThread.join();
    }
}

void lowLevelControl::robotCartesianMove()
{
    auto callbackFun =
        [this](const franka::RobotState& robot_state,
               franka::Duration time_step) -> franka::JointVelocities
    {
        //  RT Control Loop
        this->mutex.lock();
        this->time += time_step.toSec();

        this->J_zeroJac = Eigen::Map<const Eigen::Matrix<double, 6, 7>>(
            this->pandaModel
                .zeroJacobian(franka::Frame::kEndEffector, robot_state)
                .data());

        if (this->callbackRobotState)
        {
            (*this->callbackRobotState)(robot_state, this->time);
        }

        // Args work, call doesn't work
        this->nextVelocities = this->_inverseKinematics->calculateIK(
            Eigen::Map<const Eigen::Matrix<double, 7, 1>>(robot_state.q.data()),
            this->_trajectory.getVelocity(this->time).transpose(),
            this->J_zeroJac);

        bool exitLoop = false;

        if (this->_trajectory.getVelocity(this->time).norm() == 0.0)
        {
            this->absVelocity =
                abs(robot_state.dq[0]) + abs(robot_state.dq[1]) +
                abs(robot_state.dq[2]) + abs(robot_state.dq[3]) +
                abs(robot_state.dq[4]) + abs(robot_state.dq[5]) +
                abs(robot_state.dq[6]);

            // std::cout << "Vel: " << this->absVelocity << "\n";

            // Trajectory Finished. Check for velocities to be 0
            if (this->absVelocity < 0.02)
            {
                exitLoop = true;
            }
        }

        Eigen::VectorXd::Map(&this->jointVels[0], 7) = this->nextVelocities;
        this->mutex.unlock();

        if (exitLoop)
        {
            // Add Stopping Criteria
            return franka::MotionFinished(
                franka::JointVelocities({0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}));
        }
        else
        {
            return franka::JointVelocities(this->jointVels);
        }
    };
    try
    {
        this->_robot.control(callbackFun);
        std::cout << "Finished Trajectory.\n" << std::endl;
    }
    catch (const franka::Exception& e)
    {
        std::cout << "Robot Control Loop failed.\n";
        std::cerr << e.what() << std::endl;
        this->robotStatus = Status::idle;
    }
}

void lowLevelControl::registerTrajectoryCartesianCallback(
    std::shared_ptr<motion> _motion)
{
    std::this_thread::sleep_for(std::chrono::milliseconds(5));

    bool tryagain = true;
    while (tryagain)
    {
        {
            if (mutex.try_lock())
            {

                trajectoryQueue.push_back(_motion);

                std::cout << "Queue Size: " << trajectoryQueue.size() << "\n";
                mutex.unlock();
                break;
            }
        }
    }
}

void lowLevelControl::registerTrajectoryJointCallback(
    std::shared_ptr<motion> _motion)
{
    std::this_thread::sleep_for(std::chrono::milliseconds(5));

    bool tryagain = true;
    while (tryagain)
    {
        {
            if (mutex.try_lock())
            {

                trajectoryQueueJointSpace.push_back(_motion);

                std::cout << "Queue Size: " << trajectoryQueueJointSpace.size()
                          << "\n";
                mutex.unlock();
                break;
            }
        }
    }
}

void lowLevelControl::addJointControl(Eigen::MatrixXd jointVelocityAdd)
{
    bool tryagain = true;
    while (tryagain)
    {
        {
            if (mutex.try_lock())
            {
                _jointVelocityAdd = jointVelocityAdd;
                mutex.unlock();
                break;
            }
        }
    }
}

void lowLevelControl::addCartVelocity(Eigen::MatrixXd cartVelocityAdd)
{
    bool tryagain = true;
    while (tryagain)
    {
        {
            if (mutex.try_lock())
            {
                _cartVelocityAdd = cartVelocityAdd;
                mutex.unlock();
                std::cout << "Added Velocity\n"
                          << cartVelocityAdd.transpose() << "\n";
                break;
            }
        }
    }
}

void lowLevelControl::changeImpedance(Eigen::MatrixXd new_k_gains,
                                      Eigen::MatrixXd new_d_gains,
                                      double new_factorAtan)
{
    bool tryagain = true;
    while (tryagain)
    {
        {
            if (mutex.try_lock())
            {
                d_gains = new_d_gains;
                k_gains = new_k_gains;
                factor = new_factorAtan;
                mutex.unlock();
                break;
            }
        }
    }
}

void lowLevelControl::registerRobotStateCallback(
    std::function<void(const franka::RobotState&, double&)>* _callbackRoboState)
{
    if (mutex.try_lock())
    {
        callbackRobotState = _callbackRoboState;
        mutex.unlock();
    }
    else
    {
        std::cout << "Can't register Robot State Callback. Mutex failed.\n";
    }
}

Eigen::VectorXd lowLevelControl::lowPassFilter(Eigen::VectorXd newInput,
                                               Eigen::VectorXd oldInput,
                                               double cutOffFrequency,
                                               double timestep)
{
    auto gain = timestep / (timestep + 1 / (2 * M_PI * cutOffFrequency));
    return gain * newInput + (1 - gain) * oldInput;
}
