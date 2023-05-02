#pragma once
#include <array>

struct trajectoryJointSpace
{
    bool active = true;
    std::array<double, 7> q;
    std::array<double, 7> dq;
    std::array<double, 7> ddq;
};

struct trajectoryCartesianSpace
{
    bool active = true;
    std::array<double, 16> pose;
    std::array<double, 6> v;
    std::array<double, 6> a;
};
