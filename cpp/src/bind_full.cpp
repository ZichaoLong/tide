// Diagnostic client adapter. Core Full/row projection has no Python dependency.
#include "tide/row_emit.h"
#include "tide/isolated_linear.h"
#include <torch/csrc/utils/pybind.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace tide;
void bind_full(py::module_& m) {
  m.def("isolated_linear", &isolated_linear, py::call_guard<py::gil_scoped_release>());
  m.def("row_emit_probe", [](NodeWeights w, const std::vector<Tensor>& rows, const std::vector<Index>& times,
                            const std::vector<Index>& logical, const std::vector<Index>& phases,
                            Index period, Index targets, const std::string& policy) {
    if (rows.size() != times.size() || rows.empty()) throw std::invalid_argument("row/time mismatch");
    Graph g; g.nodes.resize(1); g.outgoing_ports.offsets = {0, static_cast<Index>(logical.size())};
    w.full_kernel = make_row_emit(logical, phases, period, targets);
    w.full_kernel->validate_weights(w, logical.size());
    Model model; model.nodes.push_back(w);
    Options opts; opts.packed = policy != "scalar"; opts.full_autograd = policy == "scalar" ? "replay" : policy;
    validate_full_autograd(model, opts);
    std::vector<Event> events; std::vector<size_t> ids;
    for (size_t i = 0; i < rows.size(); ++i) {
      Event e; e.node = 0; e.time = times[i]; e.comparison_state.value = rows[i];
      e.content = rows[i]; e.control = at::ones({}, rows[i].options());
      ids.push_back(i); events.push_back(std::move(e));
    }
    evaluate_full(g, model, events, ids, opts, opts.packed);
    std::vector<std::pair<Tensor, std::map<Index, Tensor>>> results;
    for (const auto& e : events) {
      std::map<Index, Tensor> emissions;
      for (const auto& s : e.emitted) emissions.emplace(s.slot, s.value);
      results.emplace_back(e.full, emissions);
    }
    return results;
  }, py::call_guard<py::gil_scoped_release>());
}
