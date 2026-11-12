#pragma once
// Independent formula fixture for the concrete KV/sum-or-qry/cat/ind/context/quad
// Read-selection block. This is a test oracle, never a runtime dependency.
#include "block_read_selection.h"

namespace tide::device_online::block_selection_check {
constexpr int64_t Frames=4,Nodes=3,Rows=Frames*Nodes,Banks=2,Cache=2;
constexpr int64_t Clock=(int64_t{1}<<54)+17;
using Fields=std::map<std::string,at::Tensor>;
struct Fixture {
  weighted::KVConfig kv;
  weighted::RegionalConfig regional;
  SettleBlockPlan plan;
  WeightedRawLayout raw_layout;
  std::vector<State> states;
  BlockReadSelectionInput input;
  BlockReadSelectionParameters weights;
  explicit Fixture(const std::string& memory_read="qry");
};
struct FormulaLabel {std::string kind,name;int64_t row;};
struct Formula {
  std::vector<at::Tensor> leaves;
  std::vector<FormulaLabel> labels;
  std::vector<Fields> descriptions;
  std::vector<at::Tensor> control_roots; // Before cross-region stacking: independent roots.
  std::vector<at::Tensor> memory_roots,total_roots; // Before cross-row/channel stacking.
  at::Tensor memory,total,controls;
};
at::Tensor values(at::IntArrayRef,float shift);
std::vector<at::Tensor> roles(const WeightedHeadParameters&);
std::vector<at::Tensor> roles(const WeightedKVQueryParameters&);
Formula formula(const Fixture&);
}
