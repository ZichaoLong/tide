#pragma once
#include "bounded.h"

namespace accelerator_scale::bounded {
struct Oracle { Result result;std::vector<Tensor> logits,inputs,leaves; };
void close(const Tensor&,const Tensor&,double rtol,double atol,const std::string&);
pdg_scale::Fixture fixture(pdg_scale::Config,const pdg_scale::Topology&,bool candidate);
Oracle oracle(pdg_scale::Fixture&,const pdg_scale::Config&,const pdg_scale::Topology&,const Tensor& ids);
void check_forward(pdg_scale::Config,const pdg_scale::Topology&,at::Device,Index devices);
void check_inactive_full(pdg_scale::Config,const pdg_scale::Topology&,at::Device,Index devices);
void check_training(pdg_scale::Config,const pdg_scale::Topology&,at::Device,Index devices,bool replay);
void check_replay(pdg_scale::Config,const pdg_scale::Topology&,at::Device,Index devices);
void check_peer(at::Device,at::ScalarType);
}  // namespace accelerator_scale::bounded
