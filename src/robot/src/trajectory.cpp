#include "trajectory.h"
#include "motionProfile.h"
#include "trajectoryParameters.h"
#include <iostream>

trajectory::trajectory(std::shared_ptr<motionProfile> mProfileTranslation,

                       std::shared_ptr<motionProfile> mProfileRotation,

                       Eigen::Affine3d startPose, Eigen::Affine3d goalPose,
                       trajectoryParameters motionParams, bool verbose)
    : _verbose(verbose), _startPose(startPose), _goalPose(goalPose),
      _motionProfile(std::move(mProfileTranslation)),
      _motionProfileRotation(std::move(mProfileRotation))

{
    if (_verbose)
    {
        printTrajectoryParameters(motionParams);
    }
    cartesian = true;

    Eigen::Vector3d position_initial(startPose.translation());
    Eigen::Vector3d position_desired(goalPose.translation());

    auto length_t = (position_desired - position_initial).norm();

    Eigen::Vector3d direction_t;

    if (length_t == 0.0)
    {
        direction_t << 0.0, 0.0, 0.0;
    }
    else
    {
        direction_t = (position_desired - position_initial) / length_t;
    }

    direction_cartesian_translation = direction_t;

    Eigen::Quaterniond orientation_initial(startPose.linear());
    Eigen::Quaterniond orientation_desired(goalPose.linear());
    // // "difference" quaternion
    // if (orientation_desired.coeffs().dot(orientation_initial.coeffs()) < 0.0)
    // {
    //     orientation_initial.coeffs() << -orientation_initial.coeffs();
    // }

    // "difference" quaternion
    Eigen::Quaterniond error_quaternion(orientation_initial *
                                        orientation_desired.inverse());

    if (Eigen::AngleAxisd(error_quaternion).angle() > M_PI / 2.0)
    {

        orientation_desired.coeffs() << -orientation_desired.coeffs();
        Eigen::Quaterniond error_quaternion(orientation_initial *
                                            orientation_desired.inverse());
    }
    if (Eigen::AngleAxisd(error_quaternion).angle() < M_PI / 2.0)
    {

        orientation_desired.coeffs() << -orientation_desired.coeffs();
        Eigen::Quaterniond error_quaternion(orientation_initial *
                                            orientation_desired.inverse());
    }

    // if (error_quaternion.coeffs().dot(error_quaternion.coeffs()) < 0.0)
    // {
    //     std::cout << "Changing for shortest path " << orientation_initial
    //               << "\n";

    //     error_quaternion.coeffs() << -error_quaternion.coeffs();
    // }

    // Eigen::Quaterniond error_quaternion =
    //     orientation_initial *
    //     orientation_initial.slerp(1.0, orientation_desired);

    // std::cout << "Ori ini " << orientation_initial << "\n";
    // std::cout << "Ori goal " << orientation_desired << "\n";
    // std::cout << "Ori error " << error_quaternion << "\n";

    length_r = Eigen::AngleAxisd(error_quaternion).angle();

    // if (length_r > M_PI)
    // {
    //     length_r -= 2.0 * M_PI;
    // }
    // else if (length_r < -M_PI)
    // {
    //     length_r += 2.0 * M_PI;
    // }

    // length_r = error_quaternion.vec().norm();

    // length_r = sqrt(std::pow(error_quaternion.x(), 2.0) +
    //                 std::pow(error_quaternion.y(), 2.0) +
    //                 std::pow(error_quaternion.z(), 2.0));

    Eigen::Vector3d direction_r_tmp;

    direction_r_tmp = -1.0 * Eigen::AngleAxisd(error_quaternion).axis();

    // direction_r_tmp << error_quaternion.x(), error_quaternion.y(),
    //     error_quaternion.z();
    Eigen::Vector3d direction_r;

    if (length_r == 0.0)
    {
        direction_r << 0.0, 0.0, 0.0;
    }
    else
    {
        direction_r = direction_r_tmp / direction_r_tmp.norm();
    }

    direction_cartesian_rotation = direction_r;

    // Calculate the trajectories if length
    Eigen::VectorXd acceleration_translation;
    Eigen::VectorXd velocity_translation;
    Eigen::VectorXd pose_translation;

    _motionProfile->initialize(length_t, motionParams.maximum_jerk_translation,
                               motionParams.maximum_acceleration_translation,
                               motionParams.maximum_velocity_translation);

    if (length_t == 0.0)
    {
        acceleration_translation = (Eigen::VectorXd(1) << 0.0).finished();
        velocity_translation = (Eigen::VectorXd(1) << 0.0).finished();
        pose_translation = (Eigen::VectorXd(1) << 0.0).finished();
    }
    else
    {
        acceleration_translation =
            _motionProfile->getAccelerationProfile(motionParams.timestep);

        velocity_translation =
            _motionProfile->getVelocityProfile(motionParams.timestep);

        pose_translation =
            _motionProfile->getPositionProfile(motionParams.timestep);
    }

    // Calculate the trajectories if length
    Eigen::VectorXd acceleration_rotation;
    Eigen::VectorXd velocity_rotation;
    Eigen::VectorXd pose_rotation;

    _motionProfileRotation->initialize(
        length_r, motionParams.maximum_jerk_rotation,
        motionParams.maximum_acceleration_rotation,
        motionParams.maximum_velocity_rotation);

    if (length_r == 0.0)
    {
        acceleration_rotation = (Eigen::VectorXd(1) << 0.0).finished();
        velocity_rotation = (Eigen::VectorXd(1) << 0.0).finished();
        pose_rotation = (Eigen::VectorXd(1) << 0.0).finished();
    }
    else
    {
        acceleration_rotation = _motionProfileRotation->getAccelerationProfile(
            motionParams.timestep);
        velocity_rotation =
            _motionProfileRotation->getVelocityProfile(motionParams.timestep);
        pose_rotation =
            _motionProfileRotation->getPositionProfile(motionParams.timestep);
    }

    if (velocity_translation.size() > velocity_rotation.size())
    {
        accelerations = Eigen::MatrixXd::Zero(velocity_translation.size(), 6);
        velocities = Eigen::MatrixXd::Zero(velocity_translation.size(), 6);
        positions.resize(velocity_translation.size());
    }
    else
    {
        accelerations = Eigen::MatrixXd::Zero(velocity_rotation.size(), 6);
        velocities = Eigen::MatrixXd::Zero(velocity_rotation.size(), 6);
        positions.resize(velocity_rotation.size());
    }

    accelerations.topRows(acceleration_translation.size()).leftCols(3) =
        (direction_t * acceleration_translation.transpose()).transpose();

    accelerations.topRows(acceleration_rotation.size()).rightCols(3) =
        (direction_r * acceleration_rotation.transpose()).transpose();

    velocities.topRows(velocity_translation.size()).leftCols(3) =
        (direction_t * velocity_translation.transpose()).transpose();

    velocities.topRows(velocity_rotation.size()).rightCols(3) =
        (direction_r * velocity_rotation.transpose()).transpose();

    for (size_t i = 0; i < positions.size(); i++)
    {
        Eigen::Affine3d temp = Eigen::Affine3d::Identity();
        temp.setIdentity(); // Set to Identity to make bottom row of Matrix
                            // 0,0,0,1
        double t;

        if (length_r < 1.0 / 1000.0)
        {
            t = 0.0;
        }
        else
        {
            if (i > pose_rotation.size() - 1)
            {

                t = pose_rotation(pose_rotation.size() - 1) / length_r;
            }
            else
            {

                t = pose_rotation(i) / length_r;
            }
        }

        auto rotQuat = orientation_initial.slerp(t, orientation_desired);

        temp.linear() = rotQuat.toRotationMatrix();

        if (i > pose_translation.size() - 1)
        {
            temp.translation() =
                position_initial +
                pose_translation(pose_translation.size() - 1) * direction_t;
        }
        else
        {
            temp.translation() =
                position_initial + pose_translation(i) * direction_t;
        }

        positions[i] = temp;
    }

    if (_verbose)
    {
        std::cout << "\nTrajectory Parameters:\n\t"
                  << "Length translation: " << length_t << "\n\t"
                  << "Length rotation: " << length_r << "\n\t"
                  << "Direction x y z: " << direction_t.transpose() << "\n\t"
                  << "Direction a b c: " << direction_r.transpose() << "\n\t"
                  << "Size of vector vel Profile trans: "
                  << velocity_translation.size() << "\n\t"
                  << "Size of vector vel Profile rot: "
                  << velocity_rotation.size() << "\n\t" << std::endl;
    }
}

Eigen::VectorXd trajectory::getVelocity(double& time)
{
    if (cartesian)
    {
        try
        {
            Eigen::VectorXd tempVelocities(6);
            tempVelocities.topRows(3) = _motionProfile->getVelocity(time) *
                                        direction_cartesian_translation;
            tempVelocities.bottomRows(3) =
                _motionProfileRotation->getVelocity(time) *
                direction_cartesian_rotation;
            return tempVelocities;
        }
        catch (std::exception& e)
        {
            std::cerr << "Exception caught : " << e.what() << std::endl;
            return Eigen::VectorXd::Zero(6);
        }
    }
    else
    {
        return _motionProfile->getVelocity(time) * direction_jointspace;
    }
}

Eigen::MatrixXd trajectory::getCartesianPosition(double& time)
{
    if (cartesian)
    {
        Eigen::Affine3d temp = Eigen::Affine3d::Identity();
        auto pose_rotation = _motionProfileRotation->getPosition(time);
        if (length_r == 0.0)
        {
            temp.linear() = _startPose.linear();
        }
        else
        {
            auto rotQuat = Eigen::Quaterniond(_startPose.linear())
                               .slerp(pose_rotation / (length_r),
                                      Eigen::Quaterniond(_goalPose.linear()));
            temp.linear() = rotQuat.toRotationMatrix();
        }

        auto pose_translation = _motionProfile->getPosition(time);
        temp.translation() = _startPose.translation() +
                             pose_translation * direction_cartesian_translation;

        Eigen::Matrix4d Trans;
        Trans.setIdentity();
        Trans.block<3, 3>(0, 0) = temp.linear();
        Trans.block<3, 1>(0, 3) = temp.translation();
        return Trans;
    }
    else
    {
        Eigen::Matrix4d Trans;
        Trans.setIdentity();
        return Trans;
    }
}

Eigen::VectorXd trajectory::getJointPosition(double& time)
{
    if (!cartesian)
    {
        return joint_position_initial +
               _motionProfile->getPosition(time) * direction_jointspace;
    }
    else
    {
        return Eigen::MatrixXd::Zero(7, 1);
    }
}

trajectory::trajectory(std::shared_ptr<motionProfile> mProfile,
                       Eigen::MatrixXd q_start, Eigen::MatrixXd q_goal,
                       trajectoryParameters motionParams, bool verbose)
    : _verbose(verbose), _motionProfile(std::move(mProfile))

{
    if (_verbose)
    {
        printTrajectoryParameters(motionParams);
    }
    cartesian = false;
    // _motionProfile = mProfile;

    Eigen::VectorXd position_initial(q_start);
    Eigen::VectorXd position_desired(q_goal);

    joint_position_initial = position_initial;

    auto length_t = (position_desired - position_initial).norm();

    Eigen::VectorXd direction_t;

    if (length_t == 0.0)
    {

        direction_t = (Eigen::VectorXd(7) << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                          .finished();
    }
    else
    {
        direction_t = (position_desired - position_initial) / length_t;
    }
    direction_jointspace = direction_t;

    // Calculate the trajectories if length
    Eigen::VectorXd acceleration_linspace;
    Eigen::VectorXd velocity_linspace;
    Eigen::VectorXd pose_linspace;

    _motionProfile->initialize(length_t, motionParams.maximum_jerk_translation,
                               motionParams.maximum_acceleration_translation,
                               motionParams.maximum_velocity_translation);

    if (length_t == 0.0)
    {
        acceleration_linspace = (Eigen::VectorXd(1) << 0.0).finished();
        velocity_linspace = (Eigen::VectorXd(1) << 0.0).finished();
        pose_linspace = (Eigen::VectorXd(1) << 0.0).finished();
    }
    else
    {

        acceleration_linspace =
            _motionProfile->getAccelerationProfile(motionParams.timestep);

        velocity_linspace =
            _motionProfile->getVelocityProfile(motionParams.timestep);

        pose_linspace =
            _motionProfile->getPositionProfile(motionParams.timestep);
    }

    accelerations = Eigen::MatrixXd::Zero(velocity_linspace.size(), 7);

    velocities = Eigen::MatrixXd::Zero(velocity_linspace.size(), 7);

    jointPositions = Eigen::MatrixXd::Zero(velocity_linspace.size(), 7);

    accelerations =
        (direction_t * acceleration_linspace.transpose()).transpose();
    velocities = (direction_t * velocity_linspace.transpose()).transpose();

    jointPositions = (direction_t * pose_linspace.transpose()).transpose();

    jointPositions.rowwise() += position_initial.transpose();

    if (_verbose)
    {
        std::cout << "\nTrajectory Parameters:\n\t"
                  << "Length joint space: " << length_t << "\n\t"
                  << "Direction q: " << direction_t.transpose() << "\n\t"
                  << "Size of vector vel Profile: " << velocity_linspace.size()
                  << "\n\t";
    }
}

Eigen::MatrixXd trajectory::getAccelerations() { return accelerations; }

Eigen::MatrixXd trajectory::getVelocities() { return velocities; }
std::vector<Eigen::Matrix4d> trajectory::getPoses()
{

    std::vector<Eigen::Matrix4d> v;

    for (size_t i = 0; i < positions.size(); i++)
    {
        v.push_back(Eigen::Matrix4d(positions[i].matrix()));
    }
    return v;
}

void trajectory::printTrajectoryParameters(trajectoryParameters params)
{
    std::cout << "Max Jerk Translation: " << params.maximum_jerk_translation
              << "\n";
    std::cout << "Max Acc Translation: "
              << params.maximum_acceleration_translation << "\n";
    std::cout << "Max Vel Translation: " << params.maximum_velocity_translation
              << "\n";
    std::cout << "Max Jerk Rotation: " << params.maximum_jerk_rotation << "\n";
    std::cout << "Max Acc Rotation: " << params.maximum_acceleration_rotation
              << "\n";
    std::cout << "Max Vel Rotation: " << params.maximum_velocity_rotation
              << "\n";
    std::cout << "Timestep: " << params.timestep << "\n";
};