#pragma once
// Architecture §2: physical atoms map to source owners; multiplicity is int64
// incidence. Equal values/classes never establish shared numerical identity.
#include "weighted_column_sources.h"
#include <functional>
namespace tide::weighted::columns {
struct AggregateTensorRequest {
  std::vector<const Message*> sources;
  std::vector<Tensor> values;
  std::vector<Index> rows;
};
std::vector<Received> aggregate_tensor_many(const std::vector<AggregateTensorRequest>&,Index,
    const std::function<Received(const AggregateTensorRequest&)>& fallback);
}
