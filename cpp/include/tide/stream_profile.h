#pragma once
#include "tide/types.h"
#include <chrono>

namespace tide {
// Coordinator wall time, including each worker-pool barrier. Construct before
// per-tick locals so the last phase includes their destruction. No clocks or
// timing statistics are touched when disabled; no per-worker times are summed.
class StreamProfile {
 public:
  StreamProfile(bool enabled, std::map<std::string, Index>& stats)
      : enabled_(enabled), stats_(stats) {
    if (enabled_) {
      for (const auto* key : {"events", "update", "select", "full", "commit", "cleanup", "tick"})
        stats_.try_emplace(std::string("profile_")+key+"_ns", 0);
      start_ = last_ = Clock::now();
    }
  }
  void phase(const char* next) {
    if (!enabled_) return;
    const auto now = Clock::now();
    add(now); last_ = now; phase_ = next;
  }
  ~StreamProfile() {
    if (!enabled_) return;
    const auto now = Clock::now();
    add(now);
    stats_.at("profile_tick_ns") += ns(now-start_);
  }
  StreamProfile(const StreamProfile&) = delete;
  StreamProfile& operator=(const StreamProfile&) = delete;
 private:
  using Clock = std::chrono::steady_clock;
  static Index ns(Clock::duration duration) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
  }
  void add(Clock::time_point now) { stats_.at(phase_) += ns(now-last_); }
  bool enabled_;
  std::map<std::string, Index>& stats_;
  Clock::time_point start_, last_;
  const char* phase_ = "profile_events_ns";
};
} // namespace tide
