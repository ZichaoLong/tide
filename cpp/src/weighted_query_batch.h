#pragma once
// Architecture §3 / memory modules: local snapshot queries, never state replay.
#include "weighted_recurrent_detail.h"
#include "weighted_packed.h"
#include "tide/read.h"
#include "tide/full.h"

namespace tide::weighted::query {
using States=std::vector<State>;
using Times=std::vector<Index>;
struct Visible {
  std::vector<Tensor> masks,ages;
  std::map<Index,std::vector<Index>> groups;
  Visible(const States&,const Times&,const KVConfig&);
  Tensor slot(const States&,const std::vector<Index>&,const std::string&) const;
  Tensor age(const std::vector<Index>&) const;
};
Tensor weighted_read(const Tensor& logits,const Tensor& values,const Tensor& weights,const Tensor& totals);
std::array<Tensor,3> effective(const States&,const Times&,const RecurrentConfig&);
struct Features {Tensor raw,query,key,value,gate,delta;};
Features features(const NodeWeights&,const States&,const Times&,const Tensor& u,const RecurrentConfig&);
std::array<Tensor,4> memory(const NodeWeights&,const States&,const Times&,const Tensor&,const KVConfig&);
std::array<Tensor,4> memory(const NodeWeights&,const States&,const Times&,const Tensor&,const RecurrentConfig&);
std::vector<TensorFields> describe(const NodeWeights&,const std::vector<ReadInput>&,const KVConfig&,const RegionalConfig&);
std::vector<TensorFields> describe(const NodeWeights&,const std::vector<ReadInput>&,const RecurrentConfig&,const RegionalConfig&);
std::vector<FullResult> full(const NodeWeights&,const std::vector<FullInput>&,Index,const Options&,const KVConfig&);
std::vector<FullResult> full(const NodeWeights&,const std::vector<FullInput>&,Index,const Options&,const RecurrentConfig&);
}
