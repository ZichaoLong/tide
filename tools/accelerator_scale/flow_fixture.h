#pragma once
#include "bounded.h"
#include <tide/settle.h>

namespace accelerator_scale::flows {
struct Topology {
  Graph body;
  std::vector<Index> ranks;
  Index period=0;
  bool rank_aligned=false;
};
struct Fixture {
  pdg_scale::Fixture values;
  Graph body;
  Model body_model;
  std::shared_ptr<tide::SettleGraph> settle;
  Index period;
  std::vector<Index> edge_rows;
};
Topology read_topology(const std::string&);
Fixture fixture(const pdg_scale::Config&,const Topology&,bool row_emission=true);
bounded::Schedule schedule(const Fixture&,const bounded::Limits&,bool training);
}  // namespace accelerator_scale::flows
