// Optional untimed accounting pass. No telemetry/Trackio dependency in the core.
#include "tide/operator_work.h"
#include "tide/operator_profile.h"
#include "tide/autograd_census.h"
#include <torch/csrc/utils/pybind.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

void bind_metrics(pybind11::module_& module) {
  module.def("reset_autograd_census", &tide::autograd_census::reset);
  module.def("autograd_census_metrics", &tide::autograd_census::metrics);
  module.def("register_autograd_roots", &tide::autograd_census::graph);
  module.def("reset_stage_profile", &tide::op_profile::reset);
  module.def("stage_profile_metrics", &tide::op_profile::metrics);
  module.def("reset_work", &tide::work::reset);
  module.def("work_metrics", &tide::work::metrics);
}
