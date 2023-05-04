#include "franka/gripper.h"
#include "franka/model.h"
#include "franka/robot.h"
#include <franka/exception.h>

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>


namespace py = pybind11;

PYBIND11_MODULE(libfrankaInterface, m) {
  m.doc() = "Bindings for the libfranka c++ interface for the Franka Emika Panda FCI with Python3.";

  py::class_<franka::Duration>(m, "Duration")
      .def(py::init<>(), "New duration with 0 ms")
      .def(py::init<uint64_t>())
      .def("toSec", &franka::Duration::toSec, "Returns duration in seconds")
      .def("toMSec", &franka::Duration::toMSec, "Returns duration in milliseconds")
      .def(py::self + py::self)
      .def(py::self += py::self)
      .def(py::self - py::self)
      .def(py::self -= py::self)
      .def(py::self * uint64_t())
      .def(py::self *= uint64_t())
      .def(py::self / uint64_t())
      .def(py::self /= uint64_t())
      // TODO: left out modulo operator exposing
      .def(py::self == py::self)
      .def(py::self != py::self)
      .def(py::self < py::self)
      .def(py::self <= py::self)
      .def(py::self > py::self)
      .def(py::self >= py::self);

  py::enum_<franka::RealtimeConfig>(m, "RealtimeConfig")
      .value("kEnforce", franka::RealtimeConfig::kEnforce)
      .value("kIgnore", franka::RealtimeConfig::kIgnore)
      .export_values();

  py::enum_<franka::ControllerMode>(m, "ControllerMode")
      .value("kJointImpedance", franka::ControllerMode::kJointImpedance)
      .value("kCartesianImpedance", franka::ControllerMode::kCartesianImpedance)
      .export_values();

  py::enum_<franka::RobotMode>(m, "RobotMode")
      .value("kOther", franka::RobotMode::kOther)
      .value("kIdle", franka::RobotMode::kIdle)
      .value("kMove", franka::RobotMode::kMove)
      .value("kGuiding", franka::RobotMode::kGuiding)
      .value("kReflex", franka::RobotMode::kReflex)
      .value("kUserStopped", franka::RobotMode::kUserStopped)
      .value("kAutomaticErrorRecovery", franka::RobotMode::kAutomaticErrorRecovery)
      .export_values();

  py::class_<franka::RobotState>(m, "RobotState")
      .def(py::init<>())
      .def_readwrite("O_T_EE", &franka::RobotState::O_T_EE, "EE position in base frame (col major)")
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
      .def_readwrite("m_ee", &franka::RobotState::m_ee, "Configured mass of endeffector")
      .def_readwrite("I_ee", &franka::RobotState::I_ee,
                     "Configured rotational inertia matrix of the EE load w.r.t COM (col major)")
      .def_readwrite("F_x_Cee", &franka::RobotState::F_x_Cee,
                     "Configured COM of EE load w.r.t flange frame")
      .def_readwrite("m_load", &franka::RobotState::m_load, "Configured load mass")
      .def_readwrite("I_load", &franka::RobotState::I_load,
                     "Configured load inertia matrix w.r.t load COM (col major)")
      .def_readwrite("F_x_Cload", &franka::RobotState::F_x_Cload,
                     "Configured COM of external load w.r.t flange frame")
      .def_readwrite("m_total", &franka::RobotState::m_total, "Sum of EE and load mass")
      .def_readwrite("I_total", &franka::RobotState::I_total,
                     "Configured combined EE and load inertia matrix w.r.t COM (col major)")
      .def_readwrite("F_x_Ctotal", &franka::RobotState::F_x_Ctotal,
                     "Configured COM of combindes EE and external load w.r.t flange frame")
      .def_readwrite("elbow", &franka::RobotState::elbow, "elbow configuration")
      .def_readwrite("elbow_d", &franka::RobotState::elbow_d, "desired elbow configuration")
      .def_readwrite("elbow_c", &franka::RobotState::elbow_c, "last commanded elbow configuration")
      .def_readwrite("delbow_c", &franka::RobotState::delbow_c, "Commanded elbow velocity")
      .def_readwrite("ddelbow_c", &franka::RobotState::ddelbow_c, "Commanded elbow acceleration")
      .def_readwrite("tau_J", &franka::RobotState::tau_J,
                     "Measured link-side torque sensor signals")
      .def_readwrite("tau_J_d", &franka::RobotState::tau_J_d,
                     "Desired link-side torque sensor signals")
      .def_readwrite("dtau_J", &franka::RobotState::dtau_J,
                     "Derivative of measured link-side torque sensor signals")
      .def_readwrite("q", &franka::RobotState::q, "Measured joint position")
      .def_readwrite("q_d", &franka::RobotState::q_d, "Desired joint position")
      .def_readwrite("dq", &franka::RobotState::dq, "Measured joint velocity")
      .def_readwrite("dq_d", &franka::RobotState::dq_d, "Desired joint velocity")
      .def_readwrite("ddq_d", &franka::RobotState::ddq_d, "Desired joint acceleration")
      .def_readwrite("joint_contact", &franka::RobotState::joint_contact,
                     "Activated contacts level in each joint")
      .def_readwrite("cartesian_contact", &franka::RobotState::cartesian_contact,
                     "Activated contacts level in (x,y,z,r,p,y)")
      .def_readwrite("joint_collision", &franka::RobotState::joint_collision,
                     "Activated collision level in each joint")
      .def_readwrite("cartesian_collision", &franka::RobotState::cartesian_collision,
                     "Activated collision level in (x,y,z,r,p,y)")
      .def_readwrite("tau_ext_hat_filtered", &franka::RobotState::tau_ext_hat_filtered,
                     "Filtered external torque")
      .def_readwrite("O_F_ext_hat_K", &franka::RobotState::O_F_ext_hat_K,
                     "Estimated external wrench (force, torque) acting on stiffness frame, "
                     "expressed relative to the base frame")
      .def_readwrite("K_F_ext_hat_K", &franka::RobotState::K_F_ext_hat_K,
                     "Estimated external wrench (force, torque) acting on stiffness frame, "
                     "expressed relative to the stiffness frame")
      .def_readwrite("O_dP_EE_d", &franka::RobotState::O_dP_EE_d, "Desired EE twist in base frame")
      .def_readwrite("O_T_EE_c", &franka::RobotState::O_T_EE_c,
                     "Last commanded EE pose of motion generation in base frame(col major)")
      .def_readwrite("O_dP_EE_c", &franka::RobotState::O_dP_EE_c,
                     "Last commanded EE twist in base frame")
      .def_readwrite("O_ddP_EE_c", &franka::RobotState::O_ddP_EE_c,
                     "Last commanded EE acceleration in base frame")
      .def_readwrite("theta", &franka::RobotState::theta, "motor position")
      .def_readwrite("dtheta", &franka::RobotState::dtheta, "motor velocity")
      .def_readwrite("current_errors", &franka::RobotState::current_errors, "current error state")
      .def_readwrite("last_motion_errors", &franka::RobotState::last_motion_errors,
                     "errors that aborted the previous motion")
      .def_readwrite(
          "control_command_success_rate", &franka::RobotState::control_command_success_rate,
          "percentage of last 100 control commands that were successflly received by the robot")
      .def_readwrite("robot_mode", &franka::RobotState::robot_mode)
      .def_readwrite("time", &franka::RobotState::time,
                     "strictly monotonically increasing timestamp since robot start");

  py::enum_<franka::Frame>(m, "Frame")
      .value("kJoint1", franka::Frame::kJoint1)
      .value("kJoint2", franka::Frame::kJoint2)
      .value("kJoint3", franka::Frame::kJoint3)
      .value("kJoint4", franka::Frame::kJoint4)
      .value("kJoint5", franka::Frame::kJoint5)
      .value("kJoint6", franka::Frame::kJoint6)
      .value("kJoint7", franka::Frame::kJoint7)
      .value("kFlange", franka::Frame::kFlange)
      .value("kEndEffector", franka::Frame::kEndEffector)
      .value("kStiffness", franka::Frame::kStiffness)
      .export_values();

  py::class_<franka::CartesianPose>(m, "CartesianPose")
      .def(py::init<const std::array<double, 16> &>(), py::arg("cartesian_pose"))
      // TODO: add other constructors as well
      .def("hasElbow", &franka::CartesianPose::hasElbow,
           "Determines whether elbow configuration is specified")
      .def_readwrite("O_T_EE", &franka::CartesianPose::O_T_EE,
                     "Homogeneous transformation transforming EE to base frame O (col. major)")
      .def_readwrite("elbow", &franka::CartesianPose::elbow, "Elbow configuration");

  py::class_<franka::CartesianVelocities>(m, "CartesianVelocities")
      .def(py::init<const std::array<double, 6> &>(), py::arg("cartesian_velocities"))
      // TODO: add other constructors
      .def("hasElbow", &franka::CartesianVelocities::hasElbow,
           "Determines whether elbow configuration is specified")
      .def_readwrite("O_dP_EE", &franka::CartesianVelocities::O_dP_EE,
                     "Desired Cartesian velocity w.r.t. O-frame {dx in [m/s], dy in [m/s], dz in "
                     "[m/s], omegax in [rad/s], omegay in [rad/s], omegaz in [rad/s]}.")
      .def_readwrite("elbow", &franka::CartesianVelocities::elbow, "Elbow configuration");

  py::class_<franka::JointPositions>(m, "JointPositions")
      .def(py::init<const std::array<double, 7> &>(), py::arg("joint_positions"))
      .def_readwrite("q", &franka::JointPositions::q, "Desired joint angles [rad]");

  py::class_<franka::JointVelocities>(m, "JointVelocities")
      .def(py::init<const std::array<double, 7> &>(), py::arg("joint_velocities"))
      .def_readwrite("dq", &franka::JointVelocities::dq, "Desired joint angle velocities [rad/s]");

  py::class_<franka::Torques>(m, "Torques")
      .def(py::init<const std::array<double, 7> &>(), py::arg("torques"))
      .def_readwrite("tau_J", &franka::Torques::tau_J, "Desired torques in [Nm]. ");

  m.def("MotionFinished", py::overload_cast<franka::CartesianPose>(&franka::MotionFinished),
        py::arg("cartesian_pose"));
  m.def("MotionFinished", py::overload_cast<franka::CartesianVelocities>(&franka::MotionFinished),
        py::arg("cartesian_velocities"));
  m.def("MotionFinished", py::overload_cast<franka::JointPositions>(&franka::MotionFinished),
        py::arg("joint_positions"));
  m.def("MotionFinishedJointVel",
        py::overload_cast<franka::JointVelocities>(&franka::MotionFinished),
        py::arg("joint_velocities"));
  m.def("MotionFinished", py::overload_cast<franka::Torques>(&franka::MotionFinished),
        py::arg("torques"));

  py::class_<franka::Model>(m, "Model")
      .def(
          "pose",
          +[](const franka::Model *self, franka::Frame frame, const franka::RobotState &state)
              -> std::array<double, 16> { return self->pose(frame, state); },
          "Get the 4x4 robot pose in base frame (row major)", py::arg("frame"), py::arg("state"))
      .def(
          "bodyJacobian",
          +[](const franka::Model *self, franka::Frame frame, const franka::RobotState &state)
              -> std::array<double, 42> { return self->bodyJacobian(frame, state); },
          "Get the 6x7 Jacobian for the given frame, relative to that frame (row major)",
          py::arg("frame"), py::arg("state"))
      .def(
          "zeroJacobian",
          +[](const franka::Model *self, franka::Frame frame, const franka::RobotState &state)
              -> std::array<double, 42> { return self->zeroJacobian(frame, state); },
          "Get the 6x7 Jacobian for the given joint, relative to the base frame (row major)",
          py::arg("frame"), py::arg("state"))
      .def(
          "mass",
          +[](const franka::Model *self, const franka::RobotState &state)
              -> std::array<double, 49> { return self->mass(state); },
          "Get the 7x7 mass matrix (row major)", py::arg("state"))
      .def(
          "coriolis",
          +[](const franka::Model *self, const franka::RobotState &state) -> std::array<double, 7> {
            return self->coriolis(state);
          },
          "Get the 7x1 coriolis vector", py::arg("state"));
  //   .def(
  //       "gravity",
  //       +[](const franka::Model *self, const franka::RobotState &state
  //           /*const std::array<double, 3> &gravity_earth*/) -> std::array<double, 7> {
  //         return self->gravity(state);
  //       },
  //       "Get the 7x1 gravity vector", py::arg("state"));
  // TODO: implement interface based on matrices additional to frame + RobotState

  py::class_<franka::GripperState>(m, "GripperState")
      .def(py::init<>())
      .def_readwrite("width", &franka::GripperState::width, "Current opening width")
      .def_readwrite("max_width", &franka::GripperState::max_width, "Max opening width")
      .def_readwrite("is_grasped", &franka::GripperState::is_grasped,
                     "Whether an object is grasped")
      .def_readwrite("temperature", &franka::GripperState::temperature,
                     "Current gripper temperature")
      .def_readwrite("time", &franka::GripperState::time,
                     "strictly monotonically increasing timestamp since robot start");

  py::class_<franka::Gripper>(m, "Gripper")
      .def(py::init<const std::string &>(), py::arg("franka_address"))
      .def("homing", &franka::Gripper::homing, "Performs homing of the gripper")
      .def("grasp", &franka::Gripper::grasp, "Grasps an object with defined width",
           py::arg("width"), py::arg("speed"), py::arg("force"), py::arg("epsilon_inner") = 0.005,
           py::arg("epsilon_outer") = 0.005)
      .def("move", &franka::Gripper::move, "Moves the gripper's gingers to a specified width",
           py::arg("width"), py::arg("speed"))
      .def("stop", &franka::Gripper::stop, "Stops all gripper motion")
      .def("readOnce", &franka::Gripper::readOnce, "Waits for and returns the gripper state")
      .def("serverVersion", &franka::Gripper::serverVersion,
           "Returns the software version of the server");

  py::class_<franka::Robot>(m, "Robot")
      .def(py::init<const std::string &, franka::RealtimeConfig, size_t>(),
           py::arg("franka_address"), py::arg("realtime_config") = franka::RealtimeConfig::kEnforce,
           py::arg("log_size") = 50)
      .def("read", &franka::Robot::read,
           "pass a callback with the RobotState as argument to be called cyclically.",
           py::arg("read_callback"))
      .def("readOnce", &franka::Robot::readOnce, "Wait for a robot state update and return it")
      .def("loadModel", &franka::Robot::loadModel, "Return a model instance for the robot")
      .def("serverVersion", &franka::Robot::serverVersion,
           "Returns the software version of the server")
      // Motion generation and joint-level torque commands (keep callbacks simple, must execute in
      // ~0.5 ms alltogether):
      .def("controlJointVelocities",
           py::overload_cast<
               std::function<franka::JointVelocities(const franka::RobotState &, franka::Duration)>,
               franka::ControllerMode, bool, double>(&franka::Robot::control),
           "Starts a control loop for a joint velocities motion generator with a given controller "
           "mode",
           py::arg("motion_generator_callback"),
           py::arg("controller_mode") = franka::ControllerMode::kJointImpedance,
           py::arg("limit_rate") = true,
           py::arg("cutoff_frequency") = franka::kDefaultCutoffFrequency)
      .def("controlCartesianVelocities",
           py::overload_cast<std::function<franka::CartesianVelocities(const franka::RobotState &,
                                                                       franka::Duration)>,
                             franka::ControllerMode, bool, double>(&franka::Robot::control),
           "Starts a control loop for a cartesian velocities motion generator with a given "
           "controller mode",
           py::arg("motion_generator_callback"),
           py::arg("controller_mode") = franka::ControllerMode::kJointImpedance,
           py::arg("limit_rate") = true,
           py::arg("cutoff_frequency") = franka::kDefaultCutoffFrequency)
      .def("control",
           py::overload_cast<
               std::function<franka::CartesianPose(const franka::RobotState &, franka::Duration)>,
               franka::ControllerMode, bool, double>(&franka::Robot::control),
           "Starts a control loop for a cartesian pose motion generator with a given controller "
           "mode",
           py::arg("motion_generator_callback"),
           py::arg("controller_mode") = franka::ControllerMode::kJointImpedance,
           py::arg("limit_rate") = true,
           py::arg("cutoff_frequency") = franka::kDefaultCutoffFrequency)
      .def("control",
           py::overload_cast<
               std::function<franka::Torques(const franka::RobotState &, franka::Duration)>, bool,
               double>(&franka::Robot::control),
           "Starts a control loop for sending joint-level torque commands",
           py::arg("control_callback"), py::arg("limit_rate") = true,
           py::arg("cutoff_frequency") = franka::kDefaultCutoffFrequency)
      .def("control",
           py::overload_cast<
               std::function<franka::Torques(const franka::RobotState &, franka::Duration)>,
               std::function<franka::JointPositions(const franka::RobotState &, franka::Duration)>,
               bool, double>(&franka::Robot::control),
           "starts a control loop for sending joint-level torque commands and joint positions.",
           py::arg("control_callback"), py::arg("motion_generator_callback"),
           py::arg("limit_rate") = true,
           py::arg("cutoff_frequency") = franka::kDefaultCutoffFrequency)
      .def("control",
           py::overload_cast<
               std::function<franka::Torques(const franka::RobotState &, franka::Duration)>,
               std::function<franka::JointVelocities(const franka::RobotState &, franka::Duration)>,
               bool, double>(&franka::Robot::control),
           "starts a control loop for sending joint-level torque commands and joint velocities.",
           py::arg("control_callback"), py::arg("motion_generator_callback"),
           py::arg("limit_rate") = true,
           py::arg("cutoff_frequency") = franka::kDefaultCutoffFrequency)
      .def("control",
           py::overload_cast<
               std::function<franka::Torques(const franka::RobotState &, franka::Duration)>,
               std::function<franka::CartesianPose(const franka::RobotState &, franka::Duration)>,
               bool, double>(&franka::Robot::control),
           "starts a control loop for sending joint-level torque commands and cartesian pose.",
           py::arg("control_callback"), py::arg("motion_generator_callback"),
           py::arg("limit_rate") = true,
           py::arg("cutoff_frequency") = franka::kDefaultCutoffFrequency)
      .def(
          "control",
          py::overload_cast<
              std::function<franka::Torques(const franka::RobotState &, franka::Duration)>,
              std::function<franka::CartesianVelocities(const franka::RobotState &,
                                                        franka::Duration)>,
              bool, double>(&franka::Robot::control),
          "starts a control loop for sending joint-level torque commands and cartesian velocities.",
          py::arg("control_callback"), py::arg("motion_generator_callback"),
          py::arg("limit_rate") = true,
          py::arg("cutoff_frequency") = franka::kDefaultCutoffFrequency)
      .def("control",
           py::overload_cast<
               std::function<franka::JointPositions(const franka::RobotState &, franka::Duration)>,
               franka::ControllerMode, bool, double>(&franka::Robot::control),
           "Starts a control loop for a joint position motion generator with a given controller "
           "mode",
           py::arg("motion_generator_callback"),
           py::arg("controller_mode") = franka::ControllerMode::kJointImpedance,
           py::arg("limit_rate") = true,
           py::arg("cutoff_frequency") = franka::kDefaultCutoffFrequency)

      // Commands (never call these within control or motion generators)
      // TODO: virtual wall retreival `getVirtualWall`
      .def("setCollisionBehavior",
           py::overload_cast<const std::array<double, 7> &, const std::array<double, 7> &,
                             const std::array<double, 7> &, const std::array<double, 7> &,
                             const std::array<double, 6> &, const std::array<double, 6> &,
                             const std::array<double, 6> &, const std::array<double, 6> &>(
               &franka::Robot::setCollisionBehavior),
           "Changes the collision behavior ", py::arg(" lower_torque_thresholds_acceleration "),
           py::arg("upper_torque_thresholds_acceleration"),
           py::arg("lower_torque_thresholds_nominal"), py::arg("upper_torque_thresholds_nominal"),
           py::arg("lower_force_thresholds_acceleration"),
           py::arg("upper_force_thresholds_acceleration"),
           py::arg("lower_force_thresholds_nominal"), py::arg("upper_force_thresholds_nominal"))
      .def("setCollisionBehavior",
           py::overload_cast<const std::array<double, 7> &, const std::array<double, 7> &,
                             const std::array<double, 6> &, const std::array<double, 6> &>(
               &franka::Robot::setCollisionBehavior),
           "Changes the collision behavior ", py::arg("lower_torque_thresholds"),
           py::arg("upper_torque_thresholds"), py::arg("lower_force_thresholds"),
           py::arg("upper_force_thresholds"))
      .def("setJointImpedance", &franka::Robot::setJointImpedance,
           "Sets the impedance for each joint in the internal controller", py::arg("K_theta"))
      .def(
          "setCartesianImpedance", &franka::Robot::setCartesianImpedance,
          "Sets the Cartesian impedance for (x, y, z, roll, pitch, yaw) in the internal controller",
          py::arg("K_x"))
      .def("setGuidingMode", &franka::Robot::setGuidingMode,
           "Locks or unlocks guiding mode movement in (x, y, z, roll, pitch, yaw)",
           py::arg("guiding_mode"), py::arg("elbow"))
      .def("setK", &franka::Robot::setK,
           "Sets the transformation EE_T_K from EE frame to stiffness frame (used for cartesian "
           "impedance control and for measuring/applying forces)",
           py::arg("EE_T_K"))
      .def("setEE", &franka::Robot::setEE,
           "Sets the transformation NE_T_EE from nominal endeffector to EE frame (nominal EE frame "
           "cannotbe set programatically)",
           py::arg("NE_T_EE"))
      .def("setLoad", &franka::Robot::setLoad, "Set dynamic parameters of a payload",
           py::arg("mass"), py::arg("F_x_Cload"), py::arg("load_inertia"))
      // deprecated:
      //.def("setFilters", &franka::Robot::setFilters,
      //     "Sets the cut off frequency for the given motion generator or controller",
      //     py::arg("joint_position_filter_frequency"), py::arg("joint_velocity_filter_frequency"),
      //     py::arg("cartesian_position_filter_frequency"),
      //     py::arg("cartesian_velocity_filter_frequency"), py::arg("controller_filter_frequency"))
      .def("automaticErrorRecovery", &franka::Robot::automaticErrorRecovery,
           "Runs automatic recovery from robot errors")
      .def("stop", &franka::Robot::stop,
           "Stops currently running controllers and motion generators");
}