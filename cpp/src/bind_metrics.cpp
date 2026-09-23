// Optional untimed accounting pass. No telemetry/Trackio dependency in the core.
#include "tide/operator_work.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

void bind_metrics(pybind11::module_& module) {
  module.def("reset_work", &tide::work::reset);
  module.def("work_metrics", &tide::work::metrics);
}
