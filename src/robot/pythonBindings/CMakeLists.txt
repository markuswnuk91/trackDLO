find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(PythonLibs 3)
include_directories(${PYTHON_INCLUDE_DIRS})

find_package(PythonInterp REQUIRED)
if(NOT PythonInterp_FOUND)
  message(WARNING "BUILD_PYTHON_BINDINGS is ON, but failed to find PythonInterp. "
    "Disabling pycontrolRobot."
  )
  return()
endif()

find_package(pybind11 REQUIRED)
if(NOT pybind11_FOUND)
  message(WARNING "BUILD_PYTHON_BINDINGS is ON, but failed to find pybind11. "
    "Disabling pycontrolRobot."
  )
  return()
endif()

# Build the target
# add_library(motion motion.cpp ${required_libraries})
# target_link_libraries(motion robotLib)

# Python binding module name
set(pybind_module robotBindings)

pybind11_add_module(${pybind_module} robotBindings.cpp)
set_target_properties(${pybind_module} PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
target_include_directories(${pybind_module} PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(${pybind_module} PRIVATE robotInterface ${required_libraries} ${PYTHON_LIBRARIES})

