#pragma once
#include "flow_execution.h"

namespace accelerator_scale::flows {
struct Config {
  pdg_scale::Config model;
  std::string family="pdg",flow="cpu",scheduler,optimizer="adamw",partition="locality";
  Scoring scoring{"cpu","cpu",at::kFloat};
  std::string ranking="cpu",events="cpu";
  Index devices=1,iterations=3,iteration_warmup=1,workspace_gib=1024;
  Index backward_threads=1,optimizer_threads=1;
  bool training=false;
  double runtime_setup_seconds=0.;
  ProfileConfig profile{-1,"window",""};
};
Config parse(int,char**);
Options options(const Config&,bool trace);
void validate(const Config&,at::Device,const Topology&);
void check(const Config&,const Topology&,at::Device);
void benchmark(const Config&,const Topology&,at::Device);
Fixture seeded(const Config&,const Topology&,bool rows=true);
Placement placed(Fixture&,const Config&,at::Device);
}  // namespace accelerator_scale::flows
