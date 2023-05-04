#include <pybind11/pybind11.h>

namespace py = pybind11;

void libfranka_bindings(py::module &m);
// TODO: declare other modules here

PYBIND11_MODULE(libfrankaInterface, m) {
  m.doc() = "Trajectory planning and control for Franka Emika Panda with Python3.";

  libfranka_bindings(m);
}