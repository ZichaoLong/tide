#pragma once
// Architecture §2: each source slot is a physical atom, irrespective of SrcClass.
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
