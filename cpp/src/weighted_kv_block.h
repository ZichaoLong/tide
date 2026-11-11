#pragma once
// KV §2–4: ephemeral hidden-prefix provenance, independent of serialization.
#include "weighted_query_batch.h"
#include <memory>

namespace tide::weighted::block {
struct Bank {
  State old;
  std::vector<Tensor> values,summaries,contents,guards;
  std::vector<Index> times;
};
struct View final:StateExecutionView {
  std::shared_ptr<const Bank> bank;
  Index end=0;
  State snapshot;
  bool valid(const State&)const;
};
std::shared_ptr<const View> view(const State*);
void attach(const NodeWeights&,const std::vector<State>&,const PackedSequence&,std::vector<State>&);
struct Row {State state;Index time;Tensor payload,control;};
struct Plan {
  std::vector<Row> rows;
  std::vector<std::shared_ptr<const View>> views;
  std::vector<std::shared_ptr<const Bank>> banks;
  std::vector<size_t> owner;
  Tensor indices,visible,ages,reverse;
  Index length=0;
  bool full=false;
};
std::optional<Plan> plan(const std::vector<Row>&,const KVConfig&,bool full);
Tensor attention(const Tensor&,const Tensor&,const Tensor&,const Tensor&);
TensorFields compute(const NodeWeights&,const KVConfig&,const RegionalConfig&,const Plan&,
    const std::vector<TensorFields>& old,const Tensor& values,const Tensor& summaries,
    const Tensor& contents,const Tensor& payload,const Tensor& controls);
std::vector<TensorFields> evaluate(const NodeWeights&,const KVConfig&,const RegionalConfig&,const Plan&);
std::vector<TensorFields> read(const NodeWeights&,const std::vector<ReadInput>&,const KVConfig&,
    const RegionalConfig&,bool grad,bool batched_describe);
std::vector<FullResult> full(const NodeWeights&,const std::vector<FullInput>&,Index,const Options&,
    const KVConfig&,bool grad);
}
