#include "scale.h"
#include "tide/operator_work.h"
#include "tide/operator_profile.h"
#include "../bench/streaming.h"
#include <stdexcept>

namespace pdg_scale {
namespace {
tide::Result traced(const Config& c, const Topology& topology, Index workers, bool packed,
                    const std::string& emission, bool profile = false, bool optimized = false) {
  auto local = c; local.emission = emission;
  if (!optimized) {
    local.fiber_pooling = "event"; local.fiber_cache = "cloned";
    local.projection_layout = "input"; local.attention_layout = "event";
  }
  portable_torch::seed_runtime(at::Device(at::kCPU), c.runtime.seed);
  auto f = fixture(local, topology);
  tide::Options opts; opts.workers = workers; opts.packed = packed; opts.trace = true;
  opts.full_autograd = optimized && packed ? c.full_autograd : "replay";
  opts.profile = profile;
  opts.parallel_regions = optimized && c.parallel_regions;
  opts.compact_events = optimized && c.compact_events;
  opts.defer_state_release = optimized && c.defer_state_release;
  opts.packed_sources = optimized && packed && c.packed_sources;
  opts.batch_next = optimized && packed && c.batch_next;
  tide::Streaming engine(std::move(f.graph), std::move(f.model), opts);
  tide::Continuation q; q.identity = engine.graph().identity; q.batch_size = c.batch;
  tide::StreamingCursor cursor(engine, std::move(q)); tide::Result all;
  for (Index token = 0; token < c.steps; ++token) {
    std::vector<tide::External> inputs;
    for (Index b = 0; b < c.batch; ++b)
      inputs.push_back({b, 0, token, token*(topology.layers+1), f.embedding[(token*7+b*3)%c.vocab]});
    auto step = cursor.advance(inputs, (token+1)*(topology.layers+1), (token+1)*(topology.layers+1));
    all.trace.insert(all.trace.end(), step.trace.begin(), step.trace.end());
    all.messages.insert(all.messages.end(), step.messages.begin(), step.messages.end());
    all.outputs.insert(all.outputs.end(), step.outputs.begin(), step.outputs.end());
  }
  all.continuation = cursor.snapshot(); return all;
}
}
void check(const Config& c, const Topology& t) {
  at::NoGradGuard guard;
  tide::work::reset(false);
  tide::op_profile::reset(false);
  auto expected = traced(c, t, 1, false, "slot");
  for (const auto& variant : {std::make_pair(Index{1}, false), std::make_pair(Index{1}, true), std::make_pair(Index{3}, true)}) {
    tide::work::reset(c.work_count);
    tide::op_profile::reset(c.operator_profile && variant.second);
    auto actual = traced(c, t, variant.first, variant.second, "row", c.profile, true);
    tide_bench::compare(actual, expected, true, c.runtime.dtype);
    tide::work::reset(false);
    tide::op_profile::reset(false);
  }
}
} // namespace pdg_scale
