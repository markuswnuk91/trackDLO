#pragma once

struct trajectoryParameters
{
    double maximum_jerk_translation = 6500.0 / 100.0;
    double maximum_acceleration_translation = 13.0 / 2.0;
    double maximum_velocity_translation = 0.1;
    double maximum_jerk_rotation = 12500.0 / 10.0;
    double maximum_acceleration_rotation = 25.0 / 10.0;
    double maximum_velocity_rotation = 2.5 / 3.0;
    double timestep = 1.0 / 1000.0;
};
