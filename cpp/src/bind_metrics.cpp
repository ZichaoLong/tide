// Optional untimed accounting pass. No telemetry/Trackio dependency in the core.
#include "tide/operator_work.h"
#include "tide/operator_profile.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

void bind_metrics(pybind11::module_& module) {
  module.def("reset_stage_profile", &tide::op_profile::reset);
  module.def("stage_profile_metrics", &tide::op_profile::metrics);
  module.def("reset_work", &tide::work::reset);
  module.def("work_metrics", &tide::work::metrics);
}
