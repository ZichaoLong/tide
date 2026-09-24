#pragma once
#include "portable_torch/runtime.hpp"
#include "tide/cursor.h"
#include <chrono>

namespace tide_bench {
using tide::Index;
using Clock = std::chrono::steady_clock;
inline double seconds(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now()-start).count();
}
struct Config {
  portable_torch::RuntimeOptions runtime;
  Index nodes = 32, batch = 4, width = 16, ticks = 32, rings = 4;
  Index workers = 1, warmup = 2, repetitions = 5;
  bool packed = false;
  std::string api = "cursor", run_id;
};
Config parse(int, char**);
struct Fixture { tide::Graph graph; tide::Model model; std::vector<tide::External> inputs; };
Fixture fixture(const Config&);
struct Measurement {
  tide::Result result;
  double reset = 0, advance = 0, snapshot = 0;
};
Measurement execute(tide::Streaming&, const Config&, const std::vector<tide::External>&);
void compare(const tide::Result&, const tide::Result&, bool traces, at::ScalarType payload_dtype = at::kDouble);
void check_work(const Measurement&, const Config&);
}  // namespace tide_bench
