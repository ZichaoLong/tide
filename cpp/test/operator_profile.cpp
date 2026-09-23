#include "tide/operator_profile.h"
#include <iostream>
#include <stdexcept>
#include <thread>

namespace p = tide::op_profile;
void require(bool condition) {
  if (!condition) throw std::runtime_error("exclusive profiler invariant failed");
}
int main() {
  try {
    for (bool enabled : {false, true, true, false}) {
      p::reset(enabled);
      std::array<double, 3> elapsed{};
      std::vector<std::thread> workers;
      for (int i = 0; i < 3; ++i) workers.emplace_back([&, i] {
        const auto start = std::chrono::steady_clock::now();
        for (int j = 0; j < 100; ++j) {
          p::Scope outer(p::StateOther);
          { p::Scope inner(p::Attention);
            { p::Scope nested(p::KvGather); std::this_thread::yield(); }
            inner.phase(p::Output);
          }
          outer.phase(p::StateCommit);
        }
        elapsed[i] = std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
      });
      for (auto& w : workers) w.join();
      const auto metrics = p::metrics();
      double exclusive = 0, gross = 0;
      for (auto s : elapsed) gross += s;
      for (const auto* name : p::names) {
        const auto prefix = std::string("detail/")+name;
        const auto total = metrics.at(prefix+"_worker_seconds");
        require(total >= 0 && metrics.at(prefix+"_max_seconds") <= total);
        exclusive += total;
      }
      // Summed exclusive scopes fit inside the sum of independently timed
      // thread envelopes. Nested time counted twice would violate this bound.
      require(exclusive <= gross);
      for (auto name : {"state_other", "state_commit", "attention", "kv_gather", "output"})
        require(metrics.at(std::string("detail/")+name+"_calls") == (enabled ? 300 : 0));
      require(enabled ? exclusive > 0 : exclusive == 0);
    }
    p::reset(false);
    std::cout << "PROFILE passed: nested exclusive scopes, phases, TLS, reset and disabled\n";
    return 0;
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
