#pragma once
#include "tide/types.h"

namespace tide {
// Execution choices, outside graph/checkpoint identity. "auto" takes the preset
// value; a fine switch accepts "cpu", "payload", or the exact payload device.
struct ExecutionPlacement {
  std::string preset = "native";
  std::string read = "auto", control = "auto", selection = "auto", events = "auto";
  // "profile" retains each Read contract. Explicit precision changes are only
  // available for linear Read; named norm profiles keep their declared dtype.
  std::string scoring_dtype = "profile";
};
struct ResolvedPlacement {
  at::Device payload, read, control, selection, events;
  std::string scoring_dtype;
  std::map<std::string, std::string> record() const;
};
ResolvedPlacement resolve_placement(const ExecutionPlacement&, at::Device payload);
// Shallow tensor-preserving adaptation: parameters and aliases remain owned by
// the caller. Built-in Read/Region programs only. Host schedulers require CPU
// events; a resident request is rejected rather than silently host-dispatched.
Model place_model(const Graph&, const Model&, const ExecutionPlacement&);
}  // namespace tide
