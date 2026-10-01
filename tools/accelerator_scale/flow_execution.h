#pragma once
#include "flow_fixture.h"
#include "execution.h"

namespace accelerator_scale::flows {
struct HostWindow {
  Result result;
  std::vector<Tensor> logits;
  Tensor loss;
};
HostWindow host_window(const pdg_scale::Config&,Fixture&,const Placement&,const std::string& scheduler,
                       Options,const Tensor& ids,const std::vector<Tensor>& roots={});
}  // namespace accelerator_scale::flows
