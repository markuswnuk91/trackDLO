#include "highLevelControl.h"

highLevelControl::highLevelControl(const std::string& address)
    : lowLvlCtl(address)
{

    auto success = lowLvlCtl.readOnce(&_robotState);

    stateCallback =
        [this](const franka::RobotState& robot_state,
               double& controlTime) { // memcpy(_robotState, robot_state,
                                      // sizeof(robot_state));
            this->_robotState = robot_state;
        };

    lowLvlCtl.registerRobotStateCallback(&stateCallback);
}

void highLevelControl::startTest(Eigen::Matrix4d goalPose4x4)
{
    // Initialize Impedance Control
    lowLvlCtl.startImpedanceControl();

    // Initialize Trajectory
    Eigen::Affine3d goalPose(goalPose4x4);
    Eigen::Affine3d startPose(Eigen::Matrix4d::Map(_robotState.O_T_EE.data()));

    auto cb = _motion.callbackTrajectoryCartesianSpace(startPose, goalPose);
    // lowLvlCtl.registerTrajectoryCartesianCallback(&cb);
};

std::array<double, 16> highLevelControl::get_O_T_EE()
{
    auto success = lowLvlCtl.readOnce(&_robotState);
    return _robotState.O_T_EE;
}

void highLevelControl::print_O_T_EE_and_q()
{
    auto success = lowLvlCtl.readOnce(&_robotState);

    if (success)
    {
        std::cout << "Printing Robot State\n"
                  << "q_1: " << _robotState.q[0] << "\n"
                  << "q_2: " << _robotState.q[1] << "\n"
                  << "q_3: " << _robotState.q[2] << "\n"
                  << "q_4: " << _robotState.q[3] << "\n"
                  << "q_5: " << _robotState.q[4] << "\n"
                  << "q_6: " << _robotState.q[5] << "\n"
                  << "q_7: " << _robotState.q[6] << "\n"
                  << "O_T_EE\n"
                  << "[" << _robotState.O_T_EE[0] << ","
                  << _robotState.O_T_EE[4] << "," << _robotState.O_T_EE[8]
                  << "," << _robotState.O_T_EE[12] << "]"
                  << "\n"
                  << "[" << _robotState.O_T_EE[1] << ","
                  << _robotState.O_T_EE[5] << "," << _robotState.O_T_EE[9]
                  << "," << _robotState.O_T_EE[13] << "]"
                  << "\n"
                  << "[" << _robotState.O_T_EE[2] << ","
                  << _robotState.O_T_EE[6] << "," << _robotState.O_T_EE[10]
                  << "," << _robotState.O_T_EE[14] << "]"
                  << "\n"
                  << "[" << _robotState.O_T_EE[3] << ","
                  << _robotState.O_T_EE[7] << "," << _robotState.O_T_EE[11]
                  << "," << _robotState.O_T_EE[15] << "]"
                  << "\n";
    }
    else
    {
        std::cout << "Failed getting Robot State\n";
    }
}

bool highLevelControl::check4x4Matrix(Eigen::Matrix4d mat)
{
    // Assume Translation Row Major
    double tcol = mat.coeff(3, 0) + mat.coeff(3, 1) + mat.coeff(3, 2);
    double trow = mat.coeff(0, 3) + mat.coeff(1, 3) + mat.coeff(2, 3);

    bool rowMajor;
    std::cout << "Checking Matrix\n" << mat << std::endl;

    if (trow != 0.0 && tcol == 0.0)
    {
        rowMajor = true;
        std::cout << "Found Row Major Matrix\n";
    }
    else if (trow == 0.0 && tcol != 0.0)
    {
        rowMajor = false;

        std::cout << "Found Column Major Matrix\n";
    }
    else
    {
        rowMajor = false;
        std::cout
            << "Can't see if Row/Column Major Matrix, translation = [0 0 0]\n";
    }
    return rowMajor;
}

bool highLevelControl::isIdle() { return lowLvlCtl.isIdle(); }

void highLevelControl::startImpedanceControl()
{
    if (!cartesianImpedanceIsRunning or lowLvlCtl.isIdle())
    {
        // Initialize Impedance Control
        lowLvlCtl.startImpedanceControl(lowLevelControl::impedance::Cartesian);
        cartesianImpedanceIsRunning = true;
    }
    else
    {
        std::cout << "Impedance is already running.\n";
    }
}

void highLevelControl::startJointImpedanceControl()
{
    if (!jointImpedanceIsRunning or lowLvlCtl.isIdle())
    {

        // Initialize Impedance Control
        lowLvlCtl.startImpedanceControl(lowLevelControl::impedance::JointSpace);
        jointImpedanceIsRunning = true;
    }
    else
    {
        std::cout << "Impedance is already running.\n";
    }
}

void highLevelControl::stopImpedanceControl()
{
    if (cartesianImpedanceIsRunning)
    {
        // Initialize Impedance Control
        lowLvlCtl.stopImpedanceControl();
        cartesianImpedanceIsRunning = false;
    }
    else if (jointImpedanceIsRunning)
    {

        // Initialize Impedance Control
        lowLvlCtl.stopImpedanceControl();
        jointImpedanceIsRunning = false;
    }
    else
    {
        std::cout << "Impedance is not running.\n";
    }
}

void highLevelControl::addCartVelocity(Eigen::MatrixXd cartVelocityAdd)
{
    if (cartesianImpedanceIsRunning)
    {
        // Initialize Impedance Control
        lowLvlCtl.addCartVelocity(cartVelocityAdd);
    }
    else
    {
        std::cout << "Impedance Control is not running.\n";
    }
}

void highLevelControl::addJointControl(Eigen::MatrixXd jointVelocityAdd)
{
    if (jointImpedanceIsRunning)
    {
        // Initialize Impedance Control
        lowLvlCtl.addJointControl(jointVelocityAdd);
    }
    else
    {
        std::cout << "Impedance Control is not running.\n";
    }
}

void highLevelControl::moveJointSpace(Eigen::MatrixXd q_goal,
                                      trajectoryParameters params)
{
    if (lowLvlCtl.isIdle())
    {
        this->startJointImpedanceControl();
    }

    if (jointImpedanceIsRunning)
    {
        double* q = (double*)malloc(sizeof(double) * 7);
        memcpy(q, _robotState.q.data(), sizeof(double) * 7);

        if (!initLastJointPosition)
        {
            q_start = Eigen::Matrix4d::Map(q);
            initLastJointPosition = true;
        }
        else
        {
            q_start = q_temp;
        }

        std::shared_ptr<motion> m = std::make_shared<motion>(params);

        m->initializeCallbackJointSpace(q_start, q_goal);

        q_temp = q_goal;

        lowLvlCtl.registerTrajectoryJointCallback(m);
    }
    else
    {
        std::cout << "Impedance Controller isn't running.\n";
    }
}

void highLevelControl::moveToPositionVelocity(Eigen::Matrix4d goalPose4x4,
                                              trajectoryParameters params)
{
    double* O_T_EE = (double*)malloc(sizeof(double) * 16);
    memcpy(O_T_EE, _robotState.O_T_EE.data(), sizeof(double) * 16);

    if (!initializedLastPosition)
    {
        startPose = Eigen::Matrix4d::Map(O_T_EE);
        initializedLastPosition = true;
    }
    else
    {
        startPose = goalPose;
    }

    // Initialize Trajectory
    goalPose = goalPose4x4;
    std::shared_ptr<motion> m = std::make_shared<motion>(params);

    lowLvlCtl.moveCartesian(m->getLinearTrajectory(startPose, goalPose));
}

void highLevelControl::moveToPosition(Eigen::Matrix4d goalPose4x4,
                                      trajectoryParameters params,
                                      bool emptyQueue)
{
    if (lowLvlCtl.isIdle())
    {
        this->startImpedanceControl();
    }

    if (cartesianImpedanceIsRunning)
    {
        double* O_T_EE = (double*)malloc(sizeof(double) * 16);
        memcpy(O_T_EE, _robotState.O_T_EE.data(), sizeof(double) * 16);

        if (!initializedLastPosition)
        {
            startPose = Eigen::Matrix4d::Map(O_T_EE);
            initializedLastPosition = true;
        }
        else
        {
            startPose = goalPose;
        }

        if (emptyQueue)
        {
            lowLvlCtl.emptyQueueAndStopCartesianImpedance();
            startPose = Eigen::Matrix4d::Map(O_T_EE);
        }

        // Initialize Trajectory
        goalPose = goalPose4x4;

        std::shared_ptr<motion> m = std::make_shared<motion>(params);

        m->initializeCallback(startPose, goalPose);

        lowLvlCtl.registerTrajectoryCartesianCallback(m);
    }
    else
    {
        std::cout << "Impedance Controller isn't running.\n";
    }
}

franka::RobotState highLevelControl::getRobotState()
{
    if (lowLvlCtl.isIdle())
    {
        bool getNewRobotState = lowLvlCtl.readOnce(&_robotState);
    }

    return _robotState;
}

// Gripper Functions
franka::GripperState highLevelControl::readGripper()
{
    return lowLvlCtl.readGripper();
}

void highLevelControl::homeGripper() { lowLvlCtl.homeGripper(); }

void highLevelControl::moveGripper(double width, double speed)
{
    lowLvlCtl.moveGripper(width, speed);
}

void highLevelControl::grasp(double width, double speed, double force,
                             double epsilon_inner, double epsilon_outer)
{
    lowLvlCtl.grasp(width, speed, force, epsilon_inner, epsilon_outer);
}

void highLevelControl::setImpedance(Eigen::MatrixXd new_k_gains,
                                    Eigen::MatrixXd new_d_gains,
                                    double new_factorAtan)
{
    lowLvlCtl.changeImpedance(new_k_gains, new_d_gains, new_factorAtan);
}
