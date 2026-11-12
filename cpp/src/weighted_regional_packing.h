#pragma once
// Complete regional incidence rows, grouped by real cardinality; no padding.
#include "tide/weighted_columns.h"

namespace tide::weighted::columns {
using RegionalFields = std::map<Index,TensorFields>;
struct PackedRegionalColumns {TensorFields fields;Tensor nodes;};
struct RegionalPackingGroup {
  Index candidates=0;
  std::vector<Index> rows,participants,node_ids,local_rows,gather;
  std::map<Index,std::vector<Index>> domains;
  PackedRegionalColumns pack(const RegionalFields&) const;
  // bool R or R,N structural support, independent of numerical zero bars.
  // Undefined field bars or absent source support return undefined Tensors.
  RegionalFields pullback(const RegionalFields&,const TensorFields& bars,const Tensor& support={}) const;
};
struct RegionalControlAdjoints {std::vector<Tensor> values,row_support;};
struct RegionalPacking {
  std::map<Index,std::vector<Index>> domains,inverse;
  std::vector<RegionalPackingGroup> groups;
  explicit RegionalPacking(const std::map<Index,std::vector<Index>>& rows_by_node);
  // Multi-output linear autograd boundary; one transpose per used group.
  std::map<Index,Tensor> unpack(const std::vector<Tensor>&,const Tensor& like={}) const;
  RegionalControlAdjoints controls_pullback(const std::map<Index,Tensor>&,const Tensor& like) const;
};
}
