#include "sevenPhaseProfile.h"
#include <iostream>

sevenPhaseProfile::sevenPhaseProfile(){};

void sevenPhaseProfile::initialize(double& length, double& jerk,
                                   double& acceleration, double& velocity)
{
    _length = length;
    _jerk = jerk;
    _acceleration = acceleration;
    _velocity = velocity;

    time_0_1 = 0.0;
    time_1_2 = 0.0;
    time_2_3 = 0.0;
    time_3_4 = 0.0;
    time_4_5 = 0.0;
    time_5_6 = 0.0;
    time_6_7 = 0.0;

    if (length < 0.0 || jerk < 0.0 || acceleration < 0.0 || velocity < 0.0)
    {
        throw std::invalid_argument(
            "Length, jerk, acc and velocity have to be > 0");
    }

    // Calculate the time of each Phase
    // Reach max acc
    time_0_1 = _acceleration / _jerk;

    // Reach max vel with vel after reach max acc + vel
    auto v_1 = getVelocityIncreaseInterval(_jerk, 0.0, time_0_1);
    time_1_2 = (_velocity - _acceleration * time_0_1) / _acceleration;

    // 1.0 / _acceleration * (_velocity - _jerk * std::pow(time_0_1, 2.0));

    if (time_1_2 < 0.0)
    {
        if (_verbose)
        {
            std::cout << "Max vel reached too early. Reducing maximum "
                         "acceleration.\n";
        }
        time_1_2 = 0.0;
        // time_0_1 = _velocity / _acceleration;
        time_0_1 = std::pow(_velocity / _jerk, 0.5);
        time_2_3 = time_0_1;

        _acceleration = _velocity / time_0_1;
        // time_0_1 = _acceleration / _jerk;

        if (_verbose)
        {
            std::cout << "Reduced acceleration to " << _acceleration << "\n";
        }
    }

    time_2_3 = time_0_1;
    time_4_5 = time_0_1;
    time_5_6 = time_1_2;
    time_6_7 = time_0_1;

    calculateSegmentLengths();
    // printTimes();

    if (length <= 1.0 / 10000.0)
    {
        time_0_1 = 0.0;
        time_1_2 = 0.0;
        time_2_3 = 0.0;
        time_3_4 = 0.0;
        time_4_5 = 0.0;
        time_5_6 = 0.0;
        time_6_7 = 0.0;
        if (_verbose)
        {
            std::cout << "Length < 0.1mm.\n";
            printTimes();
        }
    }
    else if (s_0_1 + s_2_3 + s_4_5 + s_6_7 > length && !(time_1_2 > 0.0))
    {

        // Cant Reach max Acc
        // Set all other times to 0
        time_1_2 = 0.0;
        time_3_4 = 0.0;
        time_4_5 = 0.0;

        // Calculate new time according to length
        // time_0_1 = std::pow(6.0 * _length / _jerk, 1.0 / 3.0);
        time_0_1 = std::pow(length / (2.0 * _jerk), 1.0 / 3.0);
        time_2_3 = time_0_1;
        time_4_5 = time_0_1;
        time_6_7 = time_0_1;
        _acceleration = _jerk * time_0_1;

        if (_verbose)
        {

            auto tmpVel =
                getVelocityIncreaseInterval(_jerk, 0.0, time_0_1) +
                getVelocityIncreaseInterval(0.0, _acceleration, time_1_2) +
                getVelocityIncreaseInterval(-_jerk, _acceleration, time_2_3);

            if (tmpVel > _velocity)
            {
                _jerk = _velocity / (time_0_1 * time_0_1);
                calculateSegmentLengths();

                time_3_4 =
                    (length - (s_0_1 + s_6_7 + s_1_2 + s_5_6 + s_2_3 + s_4_5)) /
                    _velocity;

                calculateSegmentLengths();

                std::cout
                    << "Jerk exceeds maximum velocity. Restricting max Jerk."
                    << "\n";
                std::cout << "Jerk: " << _jerk << "\n";
            }

            else
            {
                _velocity = tmpVel;
                std::cout << "Can't Reach maximum Acceleration.\n";
                std::cout << "Acceleration to: " << _acceleration << "\n";
                std::cout << "Reduced Velocity to.\n";
                std::cout << "Velocity to: " << _velocity << "\n";
            }

            calculateSegmentLengths();
            printTimes();
        }

        // Check if max vel is the constraining factor
        auto v_1 = 1.0 / 2.0 * _jerk * std::pow(time_0_1, 2.0);
        if (v_1 > _velocity)
        {
            _jerk = _velocity * 2.0 / (std::pow(time_0_1, 2.0));

            if (_verbose)
            {
                std::cout << "Max Vel constrains jerk -> reducing jerk "
                             "to "
                          << _jerk << "\n";
            }
            calculateSegmentLengths();
        }
        calculateSegmentLengths();
    }
    else if (s_0_1 + s_1_2 + s_2_3 + s_4_5 + s_5_6 + s_6_7 > length)
    {
        // printTimes();
        // If this is the case, then s_1_2 and s_5_6 have to be reduced.
        auto v_1 = getVelocityIncreaseInterval(_jerk, 0.0, time_0_1);

        auto _a = 1.0 / 2.0 * _acceleration;
        auto _b =
            1.0 / 2.0 * _jerk * time_0_1 * time_0_1 + _acceleration * time_0_1;
        auto _c = -1.0 * (length / 2.0 -
                          1.0 / 2.0 * _jerk * time_0_1 * time_0_1 * time_0_1 -
                          1.0 / 2.0 * _acceleration * time_0_1 * time_0_1);

        // s_1_2 = v_1 * t_1_2 + 1/2 * a * t_1_2^2
        auto t_1_2__a = (1.0 / (2.0 * _a)) *
                        (-_b + std::pow((_b * _b - 4.0 * _a * _c), 0.5));

        auto t_1_2__b = (1.0 / (2.0 * _a)) *
                        (-_b - std::pow((_b * _b - 4.0 * _a * _c), 0.5));

        // std::cout << "v_1: " << v_1 << "\n";
        // std::cout << "s_1_2: " << s_1_2 << "\n";

        // std::cout << "t_1_2_a: " << t_1_2__a << "\n";
        // std::cout << "t_1_2_b: " << t_1_2__b << "\n";

        // std::cout << "a: " << _a << "\n";
        // std::cout << "b: " << _b << "\n";
        // std::cout << "c: " << _c << "\n";
        // std::cout << "c*: "
        //           << -1.0 *
        //                  (length / 2.0 - _jerk * time_0_1 * time_0_1 *
        //                  time_0_1)
        //           << "\n";
        // std::cout << "j: " << _jerk << "\n";
        // std::cout << "t_0_1: " << time_0_1 << "\n";

        // std::cout << "v_1: "
        //           << getVelocityIncreaseInterval(_jerk, 0.0, time_0_1) <<
        //           "\n";
        // std::cout << "v_1_2: "
        //           << getVelocityIncreaseInterval(0.0, _acceleration,
        //           time_1_2)
        //           << "\n";
        // std::cout << "v_2_3: "
        //           << getVelocityIncreaseInterval(-_jerk, _acceleration,
        //                                          time_2_3)
        //           << "\n";

        // std::cout << "j: " << _jerk << "\n";
        // std::cout << "a: " << _acceleration << "\n";
        // std::cout << "v: " << _velocity << "\n";

        time_1_2 = t_1_2__a;

        _velocity =
            getVelocityIncreaseInterval(_jerk, 0.0, time_0_1) +
            getVelocityIncreaseInterval(0.0, _acceleration, time_1_2) +
            getVelocityIncreaseInterval(-_jerk, _acceleration, time_2_3);

        if (_verbose)
        {
            std::cout << "Can't Reach maximum Velocity. Velocity decreased to "
                      << _velocity << "\n";
        }

        // Same Jerk Phase Acc/Deacc
        time_2_3 = time_0_1;
        time_3_4 = 0.0;
        time_4_5 = time_0_1;
        time_5_6 = time_1_2;
        time_6_7 = time_0_1;

        calculateSegmentLengths();
        if (_verbose)
        {
            printTimes();
        }
    }
    else if (s_0_1 + s_6_7 + s_1_2 + s_5_6 + s_2_3 + s_4_5 < length)
    {
        // Same Jerk Phase Acc/Deacc
        time_2_3 = time_0_1;
        time_4_5 = time_0_1;
        time_5_6 = time_1_2;
        time_6_7 = time_0_1;

        calculateSegmentLengths();

        time_3_4 = (length - (s_0_1 + s_6_7 + s_1_2 + s_5_6 + s_2_3 + s_4_5)) /
                   _velocity;

        calculateSegmentLengths();

        if (_verbose)
        {
            printTimes();
        }
    }
    else
    {
        throw("Error: Failed to calculate trajectory.");
    }

    if (_verbose)
    {

        calculateSegmentLengths();
        std::cout << "** Expected Length ** \nLength: " << length << "\n";
        std::cout << "** Calculated Length ** \nLength_Trajectory: "
                  << s_0_1 + s_1_2 + s_2_3 + s_3_4 + s_4_5 + s_5_6 + s_6_7
                  << "\n";
        std::cout << "\ts_0_1: " << s_0_1 << "\n";
        std::cout << "\ts_1_2: " << s_1_2 << "\n";
        std::cout << "\ts_2_3: " << s_2_3 << "\n";
        std::cout << "\ts_3_4: " << s_3_4 << "\n";
        std::cout << "\ts_4_5: " << s_4_5 << "\n";
        std::cout << "\ts_5_6: " << s_5_6 << "\n";
        std::cout << "\ts_6_7: " << s_6_7 << "\n";
    }

    trajectory_time = time_0_1 + time_1_2 + time_2_3 + time_3_4 + time_4_5 +
                      time_5_6 + time_6_7;
}

void sevenPhaseProfile::calculateSegmentLengths()
{
    s_0_1 = getPositionIncreaseInterval(_jerk, 0.0, 0.0, time_0_1);
    auto v_0_1 = getVelocityIncreaseInterval(_jerk, 0.0, time_0_1);

    s_1_2 = getPositionIncreaseInterval(0.0, _acceleration, v_0_1, time_1_2);
    auto v_1_2 = getVelocityIncreaseInterval(0.0, _acceleration, time_1_2);

    s_2_3 = getPositionIncreaseInterval(-_jerk, _acceleration, v_0_1 + v_1_2,
                                        time_2_3);

    s_3_4 = getPositionIncreaseInterval(0.0, 0.0, _velocity, time_3_4);

    s_4_5 = getPositionIncreaseInterval(-_jerk, 0.0, _velocity, time_4_5);

    auto v_4_5 = getVelocityIncreaseInterval(-_jerk, 0.0, time_4_5);

    s_5_6 = getPositionIncreaseInterval(0.0, -_acceleration, _velocity - v_4_5,
                                        time_5_6);

    s_5_6 = s_1_2;

    auto v_5_6 = getVelocityIncreaseInterval(0.0, -_acceleration, time_5_6);

    s_6_7 = s_0_1;
}

void sevenPhaseProfile::setVerbose(bool verbose) { _verbose = verbose; }

void sevenPhaseProfile::printTimes()
{
    std::cout << "Times:\n\t"
              << "time_0_1: " << time_0_1 << "\n\t"
              << "time_1_2: " << time_1_2 << "\n\t"
              << "time_2_3: " << time_2_3 << "\n\t"
              << "time_3_4: " << time_3_4 << "\n\t"
              << "time_4_5: " << time_4_5 << "\n\t"
              << "time_5_6: " << time_5_6 << "\n\t"
              << "time_6_7: " << time_6_7 << "\n\n";
    std::cout << "\ts_0_1: " << s_0_1 << "\n";
    std::cout << "\ts_1_2: " << s_1_2 << "\n";
    std::cout << "\ts_2_3: " << s_2_3 << "\n";
    std::cout << "\ts_3_4: " << s_3_4 << "\n";
    std::cout << "\ts_4_5: " << s_4_5 << "\n";
    std::cout << "\ts_5_6: " << s_5_6 << "\n";
    std::cout << "\ts_6_7: " << s_6_7 << "\n";
}

double sevenPhaseProfile::getAcceleration(double& time)
{
    if (time < time_0_1 && time > 0.0)
    {
        // Phase 0 to 1 const Jerk
        return _jerk * time;
    }
    else if (time < time_0_1 + time_1_2 && time > 0.0)
    {
        // Phase 1 to 2 const Acc
        return _acceleration;
    }
    else if (time < time_0_1 + time_1_2 + time_2_3 && time > 0.0)
    {
        // Phase 2 to 3 const - Jerk
        auto timeInPhase = time - time_0_1 - time_1_2;

        return _acceleration - _jerk * timeInPhase;
    }
    else if (time < time_0_1 + time_1_2 + time_2_3 + time_3_4 && time > 0.0)
    {
        // Phase 3 to 4 const Vel
        return 0.0;
    }
    else if (time < time_0_1 + time_1_2 + time_2_3 + time_3_4 + time_4_5 &&
             time > 0.0)
    {
        // Phase 4 to 5 const - Jerk
        auto timeInPhase = time - time_0_1 - time_1_2 - time_2_3 - time_3_4;
        return -_jerk * timeInPhase;
    }
    else if (time < time_0_1 + time_1_2 + time_2_3 + time_3_4 + time_4_5 +
                        time_5_6 &&
             time > 0.0)
    {
        // Phase 5 to 6 const -Acc
        return -_acceleration;
    }
    else if (time < time_0_1 + time_1_2 + time_2_3 + time_3_4 + time_4_5 +
                        time_5_6 + time_6_7 &&
             time > 0.0)
    {
        // Phase 6 to 7 const Jerk
        auto timeInPhase = time - time_0_1 - time_1_2 - time_2_3 - time_3_4 -
                           time_4_5 - time_5_6;
        return -_acceleration + _jerk * timeInPhase;
    }
    else
    {
        // Out of the Trajectory time
        return 0.0;
    }
}

double sevenPhaseProfile::getVelocity(double& time)
{
    if (time < time_0_1 && time > 0.0)
    {
        // Phase 0 to 1 const Jerk
        return getVelocityIncreaseInterval(_jerk, 0.0, time);
    }
    else if (time < time_0_1 + time_1_2 && time > 0.0)
    {
        // Phase 1 to 2 const Acc
        auto timeInPhase = time - time_0_1;
        return getVelocityIncreaseInterval(_jerk, 0.0, time_0_1) +
               getVelocityIncreaseInterval(0.0, _acceleration, timeInPhase);
    }
    else if (time < time_0_1 + time_1_2 + time_2_3 && time > 0.0)
    {
        // Phase 2 to 3 const - Jerk
        auto timeInPhase = time - time_0_1 - time_1_2;
        return getVelocityIncreaseInterval(_jerk, 0.0, time_0_1) +
               getVelocityIncreaseInterval(0.0, _acceleration, time_1_2) +
               getVelocityIncreaseInterval(-_jerk, _acceleration, timeInPhase);
    }
    else if (time < time_0_1 + time_1_2 + time_2_3 + time_3_4 && time > 0.0)
    {
        // Phase 3 to 4 const Vel
        return _velocity;
    }
    else if (time < time_0_1 + time_1_2 + time_2_3 + time_3_4 + time_4_5 &&
             time > 0.0)
    {
        // Phase 4 to 5 const - Jerk
        auto timeInPhase = time - time_0_1 - time_1_2 - time_2_3 - time_3_4;
        return _velocity +
               getVelocityIncreaseInterval(-_jerk, 0.0, timeInPhase);
    }
    else if (time < time_0_1 + time_1_2 + time_2_3 + time_3_4 + time_4_5 +
                        time_5_6 &&
             time > 0.0)
    {
        // Phase 5 to 6 const -Acc
        auto timeInPhase =
            time - time_0_1 - time_1_2 - time_2_3 - time_3_4 - time_4_5;
        return _velocity + getVelocityIncreaseInterval(-_jerk, 0.0, time_4_5) +
               getVelocityIncreaseInterval(0.0, -_acceleration, timeInPhase);
    }
    else if (time < time_0_1 + time_1_2 + time_2_3 + time_3_4 + time_4_5 +
                        time_5_6 + time_6_7 &&
             time > 0.0)
    {
        // Phase 6 to 7 const Jerk
        auto timeInPhase = time - time_0_1 - time_1_2 - time_2_3 - time_3_4 -
                           time_4_5 - time_5_6;
        return _velocity + getVelocityIncreaseInterval(-_jerk, 0.0, time_4_5) +
               getVelocityIncreaseInterval(0.0, -_acceleration, time_5_6) +
               getVelocityIncreaseInterval(_jerk, -_acceleration, timeInPhase);
    }
    else
    {
        // Out of the Trajectory time
        return 0.0;
    }
}

double sevenPhaseProfile::getVelocityIncreaseInterval(double j, double a,
                                                      double time)
{
    return a * time + j / 2.0 * std::pow(time, 2.0);
}

double sevenPhaseProfile::getPositionIncreaseInterval(double j, double a,
                                                      double v, double time)
{

    return v * time + a / 2.0 * std::pow(time, 2.0) +
           j / 6.0 * std::pow(time, 3.0);
}

double sevenPhaseProfile::getPosition(double& time)
{
    if (time < time_0_1 && time > 0.0)
    {
        // Phase 0 to 1 const Jerk
        return this->getPositionIncreaseInterval(_jerk, 0, 0, time);
    }
    else if (time < time_0_1 + time_1_2 && time > 0.0)
    {
        // Phase 1 to 2 const Acc
        auto timeInPhase = time - (time_0_1);
        auto v1 = this->getVelocityIncreaseInterval(_jerk, 0.0, time_0_1);
        return this->getPositionIncreaseInterval(_jerk, 0.0, 0.0, time_0_1) +
               this->getPositionIncreaseInterval(0.0, _acceleration, v1,
                                                 timeInPhase);
    }
    else if (time < time_0_1 + time_1_2 + time_2_3 && time > 0.0)
    {
        // Phase 2 to 3 const - Jerk
        auto timeInPhase = time - (time_0_1 + time_1_2);
        auto v1 = this->getVelocityIncreaseInterval(_jerk, 0.0, time_0_1);
        auto v2 =
            this->getVelocityIncreaseInterval(0.0, _acceleration, time_1_2);

        return this->getPositionIncreaseInterval(_jerk, 0.0, 0.0, time_0_1) +
               this->getPositionIncreaseInterval(0.0, _acceleration, v1,
                                                 time_1_2) +
               this->getPositionIncreaseInterval(-_jerk, _acceleration, v1 + v2,
                                                 timeInPhase);
    }
    else if (time < time_0_1 + time_1_2 + time_2_3 + time_3_4 && time > 0.0)
    {
        // Phase 3 to 4 const Vel
        auto timeInPhase = time - (time_0_1 + time_1_2 + time_2_3);
        auto v1 = this->getVelocityIncreaseInterval(_jerk, 0.0, time_0_1);
        auto v2 =
            this->getVelocityIncreaseInterval(0.0, _acceleration, time_1_2);
        auto v3_absolute = _velocity;

        return this->getPositionIncreaseInterval(_jerk, 0.0, 0.0, time_0_1) +
               this->getPositionIncreaseInterval(0.0, _acceleration, v1,
                                                 time_1_2) +
               this->getPositionIncreaseInterval(-_jerk, _acceleration, v1 + v2,
                                                 time_2_3) +
               this->getPositionIncreaseInterval(0.0, 0.0, v3_absolute,
                                                 timeInPhase);
    }
    else if (time < time_0_1 + time_1_2 + time_2_3 + time_3_4 + time_4_5 &&
             time > 0.0)
    {
        // Phase 4 to 5 const - Jerk
        auto timeInPhase = time - (time_0_1 + time_1_2 + time_2_3 + time_3_4);
        auto v1 = this->getVelocityIncreaseInterval(_jerk, 0.0, time_0_1);
        auto v2 =
            this->getVelocityIncreaseInterval(0.0, _acceleration, time_1_2);
        auto v3_absolute = _velocity;

        return this->getPositionIncreaseInterval(_jerk, 0.0, 0.0, time_0_1) +
               this->getPositionIncreaseInterval(0.0, _acceleration, v1,
                                                 time_1_2) +
               this->getPositionIncreaseInterval(-_jerk, _acceleration, v1 + v2,
                                                 time_2_3) +
               this->getPositionIncreaseInterval(0.0, 0.0, v3_absolute,
                                                 time_3_4) +
               this->getPositionIncreaseInterval(-_jerk, 0.0, v3_absolute,
                                                 timeInPhase);
    }
    else if (time < time_0_1 + time_1_2 + time_2_3 + time_3_4 + time_4_5 +
                        time_5_6 &&
             time > 0.0)
    {
        // Phase 5 to 6 const - acc

        auto timeInPhase =
            time - (time_0_1 + time_1_2 + time_2_3 + time_3_4 + time_4_5);
        auto v1 = this->getVelocityIncreaseInterval(_jerk, 0.0, time_0_1);
        auto v2 =
            this->getVelocityIncreaseInterval(0.0, _acceleration, time_1_2);
        auto v3_absolute = _velocity;
        auto v4 = this->getVelocityIncreaseInterval(-_jerk, 0.0, time_4_5);

        return this->getPositionIncreaseInterval(_jerk, 0.0, 0.0, time_0_1) +
               this->getPositionIncreaseInterval(0.0, _acceleration, v1,
                                                 time_1_2) +
               this->getPositionIncreaseInterval(-_jerk, _acceleration, v1 + v2,
                                                 time_2_3) +
               this->getPositionIncreaseInterval(0.0, 0.0, v3_absolute,
                                                 time_3_4) +
               this->getPositionIncreaseInterval(-_jerk, 0.0, v3_absolute,
                                                 time_4_5) +
               this->getPositionIncreaseInterval(0.0, -_acceleration,
                                                 v3_absolute + v4, timeInPhase);
    }
    else if (time < time_0_1 + time_1_2 + time_2_3 + time_3_4 + time_4_5 +
                        time_5_6 + time_6_7 &&
             time > 0.0)
    {
        // Phase 6 to 7 const jerk

        auto timeInPhase = time - (time_0_1 + time_1_2 + time_2_3 + time_3_4 +
                                   time_4_5 + time_5_6);
        auto v1 = this->getVelocityIncreaseInterval(_jerk, 0.0, time_0_1);
        auto v2 =
            this->getVelocityIncreaseInterval(0.0, _acceleration, time_1_2);
        auto v3_absolute = _velocity;
        auto v4 = this->getVelocityIncreaseInterval(-_jerk, 0.0, time_4_5);
        auto v5 =
            this->getVelocityIncreaseInterval(0.0, -_acceleration, time_5_6);

        return this->getPositionIncreaseInterval(_jerk, 0.0, 0.0, time_0_1) +

               this->getPositionIncreaseInterval(0.0, _acceleration, v1,
                                                 time_1_2) +
               this->getPositionIncreaseInterval(-_jerk, _acceleration, v1 + v2,
                                                 time_2_3) +
               this->getPositionIncreaseInterval(0.0, 0.0, v3_absolute,
                                                 time_3_4) +
               this->getPositionIncreaseInterval(-_jerk, 0.0, v3_absolute,
                                                 time_4_5) +
               this->getPositionIncreaseInterval(0.0, -_acceleration,
                                                 v3_absolute + v4, time_5_6) +
               this->getPositionIncreaseInterval(
                   _jerk, -_acceleration, v3_absolute + v4 + v5, timeInPhase);
    }
    else if (time > time_0_1 + time_1_2 + time_2_3 + time_3_4 + time_4_5 +
                        time_5_6 + time_6_7 &&
             time > 0.0)
    {
        // Out of the Trajectory time
        return _length;
    }
    else
    {
        return 0.0;
    }
}

Eigen::VectorXd sevenPhaseProfile::getAccelerationProfile(double& timestepInS)
{
    auto time = 0.0;
    std::vector<double> accelerationProfile;

    while (time < trajectory_time)
    {
        accelerationProfile.push_back(this->getAcceleration(time));
        time += timestepInS;

        // std::cout << "Traj Vel: " << this->getVelocity(time) << "\n";
        // std::cout << "Curr Time: " << time << "\n\n";

        // std::cout << "Traj Time: " << trajectory_time << "\n";
        // std::cout << "Timestep: " << timestepInS << "\n";
    }
    accelerationProfile.push_back(0.0);

    return Eigen::Map<Eigen::VectorXd>(&accelerationProfile[0],
                                       accelerationProfile.size());
}

Eigen::VectorXd sevenPhaseProfile::getVelocityProfile(double& timestepInS)
{
    auto time = 0.0;
    std::vector<double> velocityProfile;

    while (time < trajectory_time)
    {
        velocityProfile.push_back(this->getVelocity(time));
        time += timestepInS;

        // std::cout << "Traj Vel: " << this->getVelocity(time) << "\n";
        // std::cout << "Curr Time: " << time << "\n\n";

        // std::cout << "Traj Time: " << trajectory_time << "\n";
        // std::cout << "Timestep: " << timestepInS << "\n";
    }
    velocityProfile.push_back(0.0);

    return Eigen::Map<Eigen::VectorXd>(&velocityProfile[0],
                                       velocityProfile.size());
}

Eigen::VectorXd sevenPhaseProfile::getPositionProfile(double& timestepInS)
{
    auto time = 0.0;
    std::vector<double> positionProfile;
    while (time < trajectory_time)
    {
        positionProfile.push_back(this->getPosition(time));
        time += timestepInS;
    }
    positionProfile.push_back(this->getPosition(time));
    return Eigen::Map<Eigen::VectorXd>(&positionProfile[0],
                                       positionProfile.size());
};
