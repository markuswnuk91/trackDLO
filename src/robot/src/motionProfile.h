#pragma once
#include <Eigen/Dense>
#include <string>
#include <vector>

class motionProfile
{
public:
    // motionProfile();
    virtual void initialize(double& length, double& jerk, double& acceleration,
                            double& velocity) = 0;
    virtual double getAcceleration(double& time) = 0;
    virtual double getPosition(double& time) = 0;
    virtual double getVelocity(double& time) = 0;
    virtual Eigen::VectorXd getVelocityProfile(double& timestepInS) = 0;
    virtual Eigen::VectorXd getPositionProfile(double& timestepInS) = 0;
    virtual Eigen::VectorXd getAccelerationProfile(double& timestepInS) = 0;
};