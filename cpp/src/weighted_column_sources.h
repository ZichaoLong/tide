#pragma once
// Architecture §2.2: one regional batch of source transforms/physical atoms.
#include "weighted_column_modules.h"
namespace tide::weighted::columns {
struct SourceRequest {Tensor content;std::vector<Tensor> parameters;double alpha,epsilon;};
struct SourceBlocks {std::vector<Tensor> values;std::vector<Index> owners;};
std::vector<Tensor> source_parameters(const NodeWeights&,Index label);
SourceBlocks source_blocks(const std::vector<SourceRequest>&,Index max_workspace_bytes);
std::vector<Tensor> source_columns(const std::vector<SourceRequest>&,Index max_workspace_bytes);
// Exact numerical aliases receive stable use-order adjoints, including zeros.
std::vector<Tensor> ordered_uses(const Tensor&,size_t count);
struct Received {std::vector<Index> rows;Tensor mass,content;};
struct AggregateRequest {
  const NodeWeights* weights;
  const NodeBinding* binding;
  std::vector<std::pair<const Message*,Index>> sources;
};
std::vector<Received> aggregate_columns_many(const std::vector<AggregateRequest>&,Index max_workspace_bytes);
}
