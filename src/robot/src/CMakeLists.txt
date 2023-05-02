find_package (Eigen3 3.3 REQUIRED NO_MODULE)
IF(Eigen3_FOUND)
  message("Found Eigen3")
ENDIF()

find_package(Franka REQUIRED)
IF(Franka_FOUND)
  message("Found Franka")
ENDIF()

find_package(Threads REQUIRED)

add_library(robotInterface lowLevelControl.cpp highLevelControl.cpp motion.cpp trajectory.cpp sevenPhaseProfile.cpp zeroSpaceInverseKinematics.cpp inverseKinematics.cpp dampedInverseKinematics.cpp)
target_include_directories(robotInterface PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(robotInterface PUBLIC Franka::Franka Eigen3::Eigen Threads::Threads)


