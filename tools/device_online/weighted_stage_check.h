#pragma once
#include "weighted_flow_vjp.h"
#include "tide/greedy.h"
namespace tide::device_online::stage_check {
constexpr int64_t rows=4,capacity=9;
struct Fixture {Graph graph;Model model;Continuation initial;std::vector<External> inputs;};
Fixture fixture(const std::string& family,const std::string& read);
AtomBatch atoms(const Fixture&,at::Device);
WeightedNodeRoots roots(const WeightedFlowNode&);
void verify(const Fixture&,const Result&,const ReadyBatch&,const WeightedFlowNodes&,const std::vector<WeightedNodeRoots>&,
  const std::map<int64_t,WeightedRowAdjoint>& controls,const std::vector<WeightedNodeGradient>&,
  const ParameterVjp&,const std::string& root);
void set_roots(const WeightedFlowNodes&,const std::vector<WeightedNodeRoots>&,
  const std::map<int64_t,WeightedRowAdjoint>& controls,const std::string& root);
}
