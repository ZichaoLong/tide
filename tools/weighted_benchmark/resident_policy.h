#pragma once
#include "benchmark.h"
#include "tide/settle.h"
#include <optional>
namespace weighted_benchmark {
struct BlockSelection {
  std::optional<SettleGraph> graph;
  Json record;
};
// Static installed bindings only. No candidate/reference trajectory or device
// allocation participates, and an admitted owner's runtime errors never retry.
BlockSelection select_resident_block(const Json&,const Fixture&,bool prefill,at::ScalarType,
  at::Device,const std::vector<at::Device>& peers,const std::vector<at::Device>& owners,const std::string& returns);
Json resident_block_limits(const Json&,const Fixture&);
}
