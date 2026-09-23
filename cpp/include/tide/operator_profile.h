#pragma once
// Optional CPU diagnostics. Reset/snapshot only with all measured workers idle.
// Exclusive elapsed time per calling thread, NOT coordinator wall time or CPU time.
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace tide::op_profile {
enum Field { InputPack, Qkv, KvBuild, KvGather, Attention, Pooling, Output,
  StateCommit, StateOther, Aggregate, Read, Next, FullOther, Norm, Emit, Fields };
inline constexpr std::array<const char*, Fields> names{
  "input_pack", "qkv", "kv_build", "kv_gather", "attention", "pooling", "output",
  "state_commit", "state_other", "aggregate", "read", "next", "full_other", "norm", "emit"};
using Count = std::int64_t;
struct Counts { std::array<Count, Fields> ns{}, calls{}, maximum{}; };
inline std::atomic<bool> active{false};
inline std::mutex mutex;
inline std::vector<std::unique_ptr<Counts>> registry;
inline thread_local Counts* local = nullptr;
inline Counts& counters() {
  if (!local) {
    std::lock_guard<std::mutex> lock(mutex);
    registry.push_back(std::make_unique<Counts>()); local = registry.back().get();
  }
  return *local;
}
inline void reset(bool enabled) {
  active.store(false, std::memory_order_relaxed);
  std::lock_guard<std::mutex> lock(mutex);
  for (auto& record : registry) *record = Counts{};
  active.store(enabled, std::memory_order_relaxed);
}
class Scope;
inline thread_local Scope* parent = nullptr;
class Scope {
  using Clock = std::chrono::steady_clock;
  bool enabled_;
  Field field_;
  Counts* counts_ = nullptr;
  Scope* parent_ = nullptr;
  Clock::time_point start_;
  Count children_ = 0;
  void record(Clock::time_point now) {
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(now-start_).count();
    const Count exclusive = elapsed-children_;
    counts_->ns[field_] += exclusive; ++counts_->calls[field_];
    if (exclusive > counts_->maximum[field_]) counts_->maximum[field_] = exclusive;
    if (parent_) parent_->children_ += elapsed;
    children_ = 0; start_ = now;
  }
 public:
  explicit Scope(Field field) : enabled_(active.load(std::memory_order_relaxed)), field_(field) {
    if (enabled_) { counts_ = &counters(); parent_ = parent; parent = this; start_ = Clock::now(); }
  }
  void phase(Field next) { if (enabled_) { record(Clock::now()); field_ = next; } }
  ~Scope() { if (enabled_) { record(Clock::now()); parent = parent_; } }
  Scope(const Scope&) = delete;
  Scope& operator=(const Scope&) = delete;
};
inline std::map<std::string, double> metrics() {
  std::lock_guard<std::mutex> lock(mutex);
  std::map<std::string, double> result;
  for (int i = 0; i < Fields; ++i) {
    Count total = 0, calls = 0, maximum = 0;
    for (const auto& record : registry) {
      total += record->ns[i]; calls += record->calls[i];
      if (record->maximum[i] > maximum) maximum = record->maximum[i];
    }
    const auto prefix = std::string("detail/")+names[i];
    result[prefix+"_worker_seconds"] = total*1e-9;
    result[prefix+"_calls"] = calls;
    result[prefix+"_max_seconds"] = maximum*1e-9;
  }
  return result;
}
} // namespace tide::op_profile
