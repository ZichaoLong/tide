#pragma once
// Optional host-metadata accounting. No tensor reads, device calls or clocks.
// One observer per process; reset/snapshot only while executor workers are idle.
#include <atomic>
#include <cstdint>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>

namespace tide::work::packing {
using Count = std::int64_t;
inline constexpr std::size_t max_keys = 4096;
inline std::atomic<bool> active{false};
inline std::mutex mutex;
inline std::map<std::string, Count> counts;
inline Count dropped_updates = 0;

inline void reset(bool on) {
  active.store(false, std::memory_order_relaxed);
  std::lock_guard<std::mutex> lock(mutex);
  counts.clear(); dropped_updates = 0;
  active.store(on, std::memory_order_relaxed);
}
inline bool enabled() { return active.load(std::memory_order_relaxed); }
inline void add_locked(const std::string& key, Count value) {
  auto found = counts.find(key);
  if (found != counts.end()) found->second += value;
  else if (counts.size() < max_keys) counts.emplace(key, value);
  else ++dropped_updates;
}
inline void bucket(const char* kind, Count length, Count rows) {
  if (!enabled()) return;
  if (length < 0 || rows < 1) throw std::invalid_argument("invalid packing bucket metadata");
  const auto base = std::string("packing/bucket/") + kind;
  std::lock_guard<std::mutex> lock(mutex);
  add_locked(base + "/calls", 1);
  add_locked(base + "/rows", rows);
  add_locked(base + "/elements", length * rows);
  add_locked(base + "/length/" + std::to_string(length) + "/rows/" + std::to_string(rows) + "/calls", 1);
}
inline void projection(const char* kind, Count rows) {
  if (!enabled()) return;
  if (rows < 0) throw std::invalid_argument("invalid projection batch metadata");
  const auto base = std::string("packing/projection/") + kind;
  std::lock_guard<std::mutex> lock(mutex);
  add_locked(base + "/calls", 1);
  add_locked(base + "/rows", rows);
  add_locked(base + "/rows/" + std::to_string(rows) + "/calls", 1);
}
inline void skipped(const char* reason, Count rows) {
  if (!enabled() || !rows) return;
  if (rows < 0) throw std::invalid_argument("invalid skipped packing metadata");
  const auto base = std::string("packing/skipped/") + reason;
  std::lock_guard<std::mutex> lock(mutex);
  add_locked(base + "/calls", 1); add_locked(base + "/rows", rows);
}
inline std::map<std::string, double> metrics() {
  std::lock_guard<std::mutex> lock(mutex);
  std::map<std::string, double> result(counts.begin(), counts.end());
  result["packing/schema_version"] = 1;
  result["packing/dropped_updates"] = dropped_updates;
  return result;
}
} // namespace tide::work::packing
