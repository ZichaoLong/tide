#include "streaming.h"
#include <algorithm>
#include <stdexcept>

namespace tide_bench {
Fixture fixture(const Config& c) {
  Fixture f; auto& g = f.graph; auto& m = f.model;
  auto o = at::TensorOptions().dtype(c.runtime.dtype).device(at::kCPU);
  tide::NodeWeights shared{at::full({c.width}, -.3, o), at::eye(c.width, o)*.1,
                          at::full({c.width}, .01, o), at::ones({c.width}, o)/c.width};
  auto one = at::ones({}, o), gain = at::full({}, .2, o);
  g.nodes.reserve(c.nodes); g.regions.reserve(c.nodes); g.edges.reserve(2*c.nodes);
  m.nodes.assign(c.nodes, shared);  // Deliberately shared weights, stated in manifest.
  for (Index v = 0; v < c.nodes; ++v) {
    g.nodes.push_back({v}); g.regions.push_back({1});
    const Index target = (v/8)*8+(v+1)%8;
    for (int duplicate = 0; duplicate < 2; ++duplicate) {
      g.edges.push_back({v, target, 1}); m.edge_scale.push_back(gain); m.agg_scale.push_back(one);
    }
  }
  for (Index ring = 0; ring < c.rings; ++ring) {
    g.inputs.push_back(8*ring); g.outputs.push_back(8*ring+7);
    m.input_scale.push_back(one); m.output_scale.push_back(one);
    for (Index b = 0; b < c.batch; ++b) {
      auto x = .1+at::sin(at::arange(c.width, o)*.13+b*.17+ring*.19+(c.runtime.seed%997)*.01)*.03;
      f.inputs.push_back({b, ring, 0, 0, x});
    }
  }
  return f;
}

Measurement execute(tide::Streaming& engine, const Config& c, const std::vector<tide::External>& inputs) {
  Measurement m; auto start = Clock::now();
  tide::Continuation q; q.identity = engine.graph().identity; q.batch_size = c.batch;
  std::unique_ptr<tide::StreamingCursor> cursor;
  if (c.api == "cursor") cursor = std::make_unique<tide::StreamingCursor>(engine, std::move(q));
  m.reset = seconds(start);
  const std::vector<tide::External> empty;
  for (Index cut = 1; cut <= c.ticks; ++cut) {
    start = Clock::now();
    tide::AdvanceResult result;
    if (cursor) result = cursor->advance(cut == 1 ? inputs : empty, cut, cut);
    else {
      auto raw = engine.run(q, cut == 1 ? inputs : empty, cut, cut);
      q = std::move(raw.continuation);
      result = {cut, std::move(raw.trace), std::move(raw.outputs), std::move(raw.messages), std::move(raw.stats)};
    }
    m.advance += seconds(start);  // Collection and comparisons are outside this interval.
    auto& all = m.result;
    all.trace.insert(all.trace.end(), result.trace.begin(), result.trace.end());
    all.messages.insert(all.messages.end(), result.messages.begin(), result.messages.end());
    all.outputs.insert(all.outputs.end(), result.outputs.begin(), result.outputs.end());
    for (const auto& [key, value] : result.stats)
      if (key != "cached_states" && key != "queued_times") all.stats[key] += value;
  }
  if (cursor) {
    start = Clock::now(); m.result.continuation = cursor->snapshot(); m.snapshot = seconds(start);
  } else m.result.continuation = std::move(q);
  return m;
}
void check_work(const Measurement& m, const Config& c) {
  auto get = [&](const std::string& key) {
    const auto found = m.result.stats.find(key); return found == m.result.stats.end() ? Index{0} : found->second;
  };
  const auto events = c.ticks*c.rings*c.batch;
  const auto calls = c.ticks*c.rings*(c.packed ? 1 : c.batch);
  if (get("candidate_events") != events || get("visited_edges") != 2*events
      || get("update_calls") != calls || get("full_calls") != calls
      || get("semantic_state_replays") || get("semantic_full_replays")
      || m.result.continuation.states.size() != static_cast<size_t>(std::min(c.ticks, Index{8})*c.rings*c.batch)
      || m.result.continuation.pending.size() != static_cast<size_t>(2*c.rings*c.batch))
    throw std::runtime_error("benchmark workload/counter invariant failed");
}
}  // namespace tide_bench
