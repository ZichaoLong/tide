#pragma once
// Batch §2 / recurrent §4: certified drivers, not a numerical prepass.
#include "weighted_detail.h"
#include "tide/packing_work.h"

namespace tide::weighted::packed {
struct Writes {
  std::vector<bool> positive;
  std::vector<Index> ids, offsets{0};
  std::map<Index,std::vector<size_t>> groups;
  Tensor index, times, receive, input;
  Writes(const std::vector<State>&,const PackedSequence&);
  Tensor group_index(Index length,const std::vector<size_t>& owners) const;
  Tensor gaps(const std::vector<State>&) const;
  PackedStates decode(const std::vector<State>&,const PackedSequence&,
                      const std::vector<std::map<std::string,Tensor>>&,Index calls) const;
};
PackedSequence independent(const std::vector<State>&,const Tensor&,const std::vector<Index>&,const ContentViews&);
Tensor matrix_scan(Tensor a,Tensor b,const Tensor& initial);
inline Tensor project(const Tensor& x,const Tensor& weight){
  work::packing::projection("weighted_mm",x.size(0));return at::matmul(weight,x.t()).t();}
struct Coefficients {
  Tensor a,b;
  bool matrix=false;
  std::vector<std::map<std::string,Tensor>> caches;
};
Coefficients gated_coefficients(const NodeWeights&,const std::vector<State>&,const Writes&,
    const Tensor& x,const Tensor& gap,const RecurrentConfig&);
PackedStates kv(const NodeWeights&,const std::vector<State>&,const PackedSequence&,const KVConfig&);
PackedStates recurrent(const NodeWeights&,const std::vector<State>&,const PackedSequence&,const RecurrentConfig&);
}
