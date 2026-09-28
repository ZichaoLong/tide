#pragma once
#include "../../cpp/scale/scale.h"

namespace accelerator_scale {
// Consumer configuration, independent of payload precision and CPU scheduling.
struct Scoring {
  std::string read_device = "cpu", control_device = "cpu";
  at::ScalarType dtype = at::kDouble;
  void validate(at::Device model_device, bool resident) const;
  std::string dtype_name() const { return dtype == at::kDouble ? "float64" : "float32"; }
};
void configure_scoring(pdg_scale::Fixture&, const Scoring&);
}  // namespace accelerator_scale
