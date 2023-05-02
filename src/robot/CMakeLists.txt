CMAKE_MINIMUM_REQUIRED(VERSION 3.8)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

project(RobotInterface
        VERSION 1
        LANGUAGES CXX)

message("----------------------------------------------------------")
message(" ${PROJECT_NAME} ${${PROJECT_NAME}_VERSION}               ")
message(" Robot Interface ")
message("")
message("Build and run it with: make")
message("----------------------------------------------------------")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Note: -O3 heavy performance
# SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread -O3 -march=native")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread -O3")

cmake_policy(SET CMP0069 NEW)

IF(CMAKE_BUILD_TYPE MATCHES DEBUG)
    message("Warning: DEBUG Mode Active!")
ENDIF(CMAKE_BUILD_TYPE MATCHES DEBUG)

# Add subdirectories
add_subdirectory(src)
add_subdirectory(pythonBindings)


