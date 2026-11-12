#pragma once
// Architecture §2: unique source owners and exact physical-use incidence.
#include "weighted_aggregate_tensor.h"
#include <cstdint>
#include <set>
namespace tide::weighted::columns {
struct AggregateSourceFootprint {
  std::set<Index> rows;
  std::set<std::tuple<std::vector<Index>,std::uintptr_t,std::uintptr_t>> sources;
  Index groups=0,width=0,element_size=0;
  void add(const AggregateTensorRequest&);
  long double bytes()const;
};
struct AggregateSourceLayout {
  Index groups=0,rows=0,sources=0,width=0;
  std::vector<Index> global_rows,lengths,physical_counts;
  std::vector<std::vector<Index>> source_rows,receiver_rows,incidence;
  std::vector<Tensor> originals,publish;
  Tensor index,inverse,counts,singleton;
  long double bytes()const;
  void materialize();
};
AggregateSourceLayout aggregate_source_layout(const std::vector<const AggregateTensorRequest*>&);
}
