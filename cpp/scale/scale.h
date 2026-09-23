#pragma once
#include "portable_torch/runtime.hpp"
#include "tide/cursor.h"
#include "tide/full.h"
#include <chrono>

namespace pdg_scale {
using tide::Index;
using Clock = std::chrono::steady_clock;
inline double seconds(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now()-start).count();
}
struct Config {
  portable_torch::RuntimeOptions runtime;
  Index width = 64, batch = 4, steps = 12, warmup = 4, workers = 1, threads = 1, vocab = 50304;
  Index head_workers = 1;
  bool packed = true, grad = false, check = false, profile = false;
  bool parallel_regions = false, compact_events = false, work_count = false, operator_profile = false;
  bool defer_state_release = false;
  std::string topology, run_id, emission = "row", attention_packing = "exact";
  std::string fiber_pooling = "event", fiber_cache = "cloned", projection_layout = "input", attention_layout = "event";
};
struct Topology {
  Index nodes, local, points, forced, budget, layers;
  std::vector<tide::Edge> edges; // Two body namespaces, logical edges, unit delays.
};
struct Fixture {
  tide::Graph graph;
  tide::Model model;
  at::Tensor embedding, head;
  std::vector<at::Tensor> owners;
  std::map<std::string, double> inventory;
};
Config parse(int, char**);
Topology read_topology(const std::string&);
Fixture fixture(const Config&, const Topology&);
std::shared_ptr<const tide::FullKernel> row_emit(std::vector<Index> logical,
                                              std::vector<Index> phases, Index period, Index targets);
void check(const Config&, const Topology&);
} // namespace pdg_scale
