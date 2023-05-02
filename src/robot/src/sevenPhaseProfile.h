#pragma once
#include "motionProfile.h"
#include <Eigen/Dense>
#include <string>
#include <vector>

class sevenPhaseProfile : public motionProfile
{
public:
    sevenPhaseProfile();
    // virtual ~sevenPhaseProfile();
    virtual void initialize(double& length, double& jerk, double& acceleration,
                            double& velocity);
    virtual double getAcceleration(double& time);
    virtual double getVelocity(double& time);
    virtual double getPosition(double& time);
    double getPositionIncreaseInterval(double j, double a, double v,
                                       double time);
    double getVelocityIncreaseInterval(double j, double a, double time);
    virtual Eigen::VectorXd getAccelerationProfile(double& timestepInS);
    virtual Eigen::VectorXd getVelocityProfile(double& timestepInS);
    virtual Eigen::VectorXd getPositionProfile(double& timestepInS);
    void printTimes();
    void setVerbose(bool verbose);

protected:
    double _length, _jerk, _acceleration, _velocity;
    double trajectory_time, time_0_1, time_1_2, time_2_3, time_3_4, time_4_5,
        time_5_6, time_6_7, s_0_1, s_1_2, s_2_3, s_3_4, s_4_5, s_5_6, s_6_7;

    void calculateSegmentLengths();
    bool _verbose = false;
};
