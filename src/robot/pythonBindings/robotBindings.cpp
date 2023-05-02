#include "../src/highLevelControl.h"
#include "../src/motion.h"
#include "../src/trajectory.h"
#include "../src/trajectoryParameters.h"
#include "../src/dampedInverseKinematics.h"
#include "../src/zeroSpaceInverseKinematics.h"

#include "franka/duration.h"
#include "franka/gripper_state.h"
#include "franka/robot_state.h"

#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

/*
Bindings Class for pycontrolRobot
*/

namespace py = pybind11;

PYBIND11_MODULE(robotBindings, m)
{
    m.doc() = "Python Bindings to test motion";

    py::class_<motion>(m, "motion")
        .def(py::init<>())
        .def(py::init<trajectoryParameters>())
        .def("setVerbosity", &motion::setVerbosity)
        .def("getLinearTrajectory",
             py::overload_cast<Eigen::Matrix4d, Eigen::Matrix4d>(
                 &motion::getLinearTrajectory),
             "Get Linear Trajectory", py::arg("Affine3d StartPose"),
             py::arg("Affine3d GoalPose"))
        .def("getLinearJointSpaceTrajectory",
             &motion::getLinearJointSpaceTrajectory);

    py::class_<trajectory>(m, "trajectory")
        .def(py::init<bool>())
        .def("getAccelerations", &trajectory::getAccelerations)
        .def("getVelocities", &trajectory::getVelocities)
        .def("getPoses", &trajectory::getPoses)
        .def("getVelocity", &trajectory::getVelocity)
        .def("getCartesianPosition", &trajectory::getCartesianPosition)
        .def("getJointPosition", &trajectory::getJointPosition)
        .def_readwrite("jointPositions", &trajectory::jointPositions)
        .def_readwrite("velocities", &trajectory::velocities)
        .def_readwrite("accelerations", &trajectory::accelerations);

    py::class_<trajectoryParameters>(m, "trajectoryParameters")
        .def(py::init<>())
        .def_readwrite("maximum_jerk_translation",
                       &trajectoryParameters::maximum_jerk_translation)
        .def_readwrite("maximum_acceleration_translation",
                       &trajectoryParameters::maximum_acceleration_translation)
        .def_readwrite("maximum_velocity_translation",
                       &trajectoryParameters::maximum_velocity_translation)
        .def_readwrite("maximum_jerk_rotation",
                       &trajectoryParameters::maximum_jerk_rotation)
        .def_readwrite("maximum_acceleration_rotation",
                       &trajectoryParameters::maximum_acceleration_rotation)
        .def_readwrite("maximum_velocity_rotation",
                       &trajectoryParameters::maximum_velocity_rotation)
        .def_readwrite("timestep", &trajectoryParameters::timestep);

    py::class_<zeroSpaceInverseKinematics>(m, "zeroSpaceInverseKinematics")
        .def(py::init<>())
        .def("calculateIK", &zeroSpaceInverseKinematics::calculateIK);

    py::class_<dampedInverseKinematics>(m, "dampedInverseKinematics")
        .def(py::init<>())
        .def("calculateIK", &dampedInverseKinematics::calculateIK);

    py::class_<highLevelControl>(m, "highLevelControl")
        .def(py::init<>())
        .def("getRobotState", &highLevelControl::getRobotState)
        .def("startImpedanceControl", &highLevelControl::startImpedanceControl)
        .def("startJointImpedanceControl",
             &highLevelControl::startJointImpedanceControl)
        .def("stopImpedanceControl", &highLevelControl::stopImpedanceControl)
        .def("isIdle", &highLevelControl::isIdle)
        .def("moveToPositionVelocity",
             &highLevelControl::moveToPositionVelocity)
        .def("moveToPosition", &highLevelControl::moveToPosition,
             py::arg("goalPose4x4"),
             py::arg("trajectoryParameters") = trajectoryParameters(),
             py::arg("emptyQueue") = false)
        // .def("moveToPosition",
        //      py::overload_cast<Eigen::Matrix4d, trajectoryParameters>(
        //          &highLevelControl::moveToPosition))
        // .def("moveToPosition",
        //      py::overload_cast<Eigen::Matrix4d, trajectoryParameters, bool>(
        //          &highLevelControl::moveToPosition))
        .def("setImpedance", &highLevelControl::setImpedance)
        .def("addJointControl", &highLevelControl::addJointControl)
        .def("addCartVelocity", &highLevelControl::addCartVelocity)
        .def("readGripper", &highLevelControl::readGripper)
        .def("homeGripper", &highLevelControl::homeGripper)
        .def("moveGripper", &highLevelControl::moveGripper)
        .def("grasp", &highLevelControl::grasp);

    py::class_<franka::GripperState>(m, "GripperState")
        .def(py::init<>())
        .def_readwrite("width", &franka::GripperState::width,
                       "Current Gripper opening Width.")
        .def_readwrite("max_width", &franka::GripperState::max_width,
                       "Maximum Gripper opening Width.")
        .def_readwrite("is_grasped", &franka::GripperState::is_grasped,
                       "Indicates whether an object is currently grasped.")
        .def_readwrite("temperature", &franka::GripperState::temperature,
                       "Current Gripper temperature.")
        .def_readwrite("time", &franka::GripperState::time,
                       "	Strictly monotonically increasing timestamp "
                       "since robot start.");

    py::class_<franka::Torques>(m, "Torques")
        .def(py::init<const std::array<double, 7>>())
        .def_readwrite("tau_J", &franka::Torques::tau_J, "Tau J");

    py::class_<franka::Duration>(m, "Duration")
        .def(py::init<>())
        .def(py::init<uint64_t>(), "Initialize with [ms]")
        .def("toSec", &franka::Duration::toSec,
             "Returns the stored duration [s]")
        .def("toMSec", &franka::Duration::toMSec,
             "Returns the stored duration [ms]");

    py::class_<franka::RobotState>(m, "RobotState")
        .def(py::init<>())
        .def_readwrite("O_T_EE", &franka::RobotState::O_T_EE,
                       "EE position in base frame (col major)")
        .def_readwrite("O_T_EE_c", &franka::RobotState::O_T_EE_c,
                       "last commanded EE position in base frame (col major)")
        .def_readwrite("O_T_EE_d", &franka::RobotState::O_T_EE_d,
                       "Last desired EE position in base frame (col major)")
        .def_readwrite("F_T_EE", &franka::RobotState::F_T_EE,
                       "EE frame pose in flange frame (col major)")
        .def_readwrite("F_T_NE", &franka::RobotState::F_T_NE,
                       "Nominal EE frame pose in flange frame (col major)")
        .def_readwrite("NE_T_EE", &franka::RobotState::NE_T_EE,
                       "EE frame pose in nominal endeffector frame (col major)")
        .def_readwrite("EE_T_K", &franka::RobotState::EE_T_K,
                       "Stiffness frame pose in EE frame (col major)")
        .def_readwrite("m_ee", &franka::RobotState::m_ee,
                       "Configured mass of endeffector")
        .def_readwrite("I_ee", &franka::RobotState::I_ee,
                       "Configured rotational inertia matrix of the EE load "
                       "w.r.t COM (col major)")
        .def_readwrite("F_x_Cee", &franka::RobotState::F_x_Cee,
                       "Configured COM of EE load w.r.t flange frame")
        .def_readwrite("m_load", &franka::RobotState::m_load,
                       "Configured load mass")
        .def_readwrite(
            "I_load", &franka::RobotState::I_load,
            "Configured load inertia matrix w.r.t load COM (col major)")
        .def_readwrite("F_x_Cload", &franka::RobotState::F_x_Cload,
                       "Configured COM of external load w.r.t flange frame")
        .def_readwrite("m_total", &franka::RobotState::m_total,
                       "Sum of EE and load mass")
        .def_readwrite(
            "I_total", &franka::RobotState::I_total,
            "Configured combined EE and load inertia matrix w.r.t COM "
            "(col major)")
        .def_readwrite("F_x_Ctotal", &franka::RobotState::F_x_Ctotal,
                       "Configured COM of combindes EE and external load w.r.t "
                       "flange frame")
        .def_readwrite("elbow", &franka::RobotState::elbow,
                       "elbow configuration")
        .def_readwrite("elbow_d", &franka::RobotState::elbow_d,
                       "desired elbow configuration")
        .def_readwrite("elbow_c", &franka::RobotState::elbow_c,
                       "last commanded elbow configuration")
        .def_readwrite("delbow_c", &franka::RobotState::delbow_c,
                       "Commanded elbow velocity")
        .def_readwrite("ddelbow_c", &franka::RobotState::ddelbow_c,
                       "Commanded elbow acceleration")
        .def_readwrite("tau_J", &franka::RobotState::tau_J,
                       "Measured link-side torque sensor signals")
        .def_readwrite("tau_J_d", &franka::RobotState::tau_J_d,
                       "Desired link-side torque sensor signals")
        .def_readwrite("dtau_J", &franka::RobotState::dtau_J,
                       "Derivative of measured link-side torque sensor signals")
        .def_readwrite("q", &franka::RobotState::q, "Measured joint position")
        .def_readwrite("q_d", &franka::RobotState::q_d,
                       "Desired joint position")
        .def_readwrite("dq", &franka::RobotState::dq, "Measured joint velocity")
        .def_readwrite("dq_d", &franka::RobotState::dq_d,
                       "Desired joint velocity")
        .def_readwrite("ddq_d", &franka::RobotState::ddq_d,
                       "Desired joint acceleration")
        .def_readwrite("joint_contact", &franka::RobotState::joint_contact,
                       "Activated contacts level in each joint")
        .def_readwrite("cartesian_contact",
                       &franka::RobotState::cartesian_contact,
                       "Activated contacts level in (x,y,z,r,p,y)")
        .def_readwrite("joint_collision", &franka::RobotState::joint_collision,
                       "Activated collision level in each joint")
        .def_readwrite("cartesian_collision",
                       &franka::RobotState::cartesian_collision,
                       "Activated collision level in (x,y,z,r,p,y)")
        .def_readwrite("tau_ext_hat_filtered",
                       &franka::RobotState::tau_ext_hat_filtered,
                       "Filtered external torque")
        .def_readwrite("O_F_ext_hat_K", &franka::RobotState::O_F_ext_hat_K,
                       "Estimated external wrench (force, torque) acting on "
                       "stiffness frame, "
                       "expressed relative to the base frame")
        .def_readwrite("K_F_ext_hat_K", &franka::RobotState::K_F_ext_hat_K,
                       "Estimated external wrench (force, torque) acting on "
                       "stiffness frame, "
                       "expressed relative to the stiffness frame")
        .def_readwrite("O_dP_EE_d", &franka::RobotState::O_dP_EE_d,
                       "Desired EE twist in base frame")
        .def_readwrite("O_T_EE_c", &franka::RobotState::O_T_EE_c,
                       "Last commanded EE pose of motion generation in base "
                       "frame(col major)")
        .def_readwrite("O_dP_EE_c", &franka::RobotState::O_dP_EE_c,
                       "Last commanded EE twist in base frame")
        .def_readwrite("O_ddP_EE_c", &franka::RobotState::O_ddP_EE_c,
                       "Last commanded EE acceleration in base frame")
        .def_readwrite("theta", &franka::RobotState::theta, "motor position")
        .def_readwrite("dtheta", &franka::RobotState::dtheta, "motor velocity")
        .def_readwrite("current_errors", &franka::RobotState::current_errors,
                       "current error state")
        .def_readwrite("last_motion_errors",
                       &franka::RobotState::last_motion_errors,
                       "errors that aborted the previous motion")
        .def_readwrite("control_command_success_rate",
                       &franka::RobotState::control_command_success_rate,
                       "percentage of last 100 control commands that were "
                       "successflly received by the robot")
        .def_readwrite("robot_mode", &franka::RobotState::robot_mode)
        .def_readwrite(
            "time", &franka::RobotState::time,
            "strictly monotonically increasing timestamp since robot start");
}