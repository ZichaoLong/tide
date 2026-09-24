// Primitive test adapter; the kernel has no Python dependency.
#include "tide/isolated_aggregate.h"
#include <torch/csrc/utils/pybind.h>
#include <pybind11/stl.h>
void bind_aggregate(pybind11::module_& m) {
  m.def("isolated_aggregate", &tide::isolated_aggregate,
        pybind11::call_guard<pybind11::gil_scoped_release>());
}
