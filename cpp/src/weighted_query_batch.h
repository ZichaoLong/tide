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
// Read consumes Z/W, H/W or H/Z/W by configuration; Full only H.
// Unused entries are undefined internal tensors, never incomplete raw states.
std::array<Tensor,3> effective(const States&,const Times&,const RecurrentConfig&,bool full=false);
struct Features {Tensor query,value,gate;};
Features features(const NodeWeights&,const States&,const Times&,const Tensor& u,const RecurrentConfig&,bool full=false);
std::array<Tensor,4> memory(const NodeWeights&,const States&,const Times&,const Tensor&,const KVConfig&,int dense=false);
std::array<Tensor,4> memory(const NodeWeights&,const States&,const Times&,const Tensor&,const RecurrentConfig&);
std::vector<TensorFields> describe(const NodeWeights&,const std::vector<ReadInput>&,const KVConfig&,const RegionalConfig&,int dense=false,bool batch_describe=false);
std::vector<TensorFields> describe(const NodeWeights&,const std::vector<ReadInput>&,const RecurrentConfig&,const RegionalConfig&,int dense=false,bool batch_describe=false);
std::vector<FullResult> full(const NodeWeights&,const std::vector<FullInput>&,Index,const Options&,const KVConfig&,int dense=false);
std::vector<FullResult> full(const NodeWeights&,const std::vector<FullInput>&,Index,const Options&,const RecurrentConfig&,int dense=false);
}
